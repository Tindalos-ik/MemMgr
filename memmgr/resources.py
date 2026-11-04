"""按配置采集主机、Python/vLLM 进程树、GPU 和 Chroma 容器资源。

主循环以单调时钟记录实际间隔；Docker 使用独立较慢周期，卷大小仅在采样开始时读取。采集失败写明状态，业务测量继续执行。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path


def sample(output: Path, interval: float, stop: threading.Event, config: dict):
    """在后台线程中将资源观测逐行写到 output/resources.jsonl，直到 stop 事件置位。

    interval 为主机目标周期，config.resources 指定容器、卷和 vLLM PID。每行记录采样起止与实际间隔；GPU
    为整卡值，磁盘 IO 为系统累计/增量，进程内存为 RSS。Docker
    采集限于指定容器，子进程超时和缺失依赖用状态字段表示。等待扣除本轮采集耗时，并允许 stop 中断。
    """
    path = output / 'resources.jsonl'
    resource_config = config.get('resources', {})
    container_name = resource_config.get('chroma_container')
    volume_name = resource_config.get('chroma_volume_name')
    volume_path = resource_config.get('chroma_volume_path')
    vllm_pids = resource_config.get('vllm_pids', [])
    docker_interval = float(resource_config.get('docker_sample_interval_seconds', 5))
    previous_start = None
    previous_disk_io = None
    next_docker = 0.0
    try:
        import psutil
        measurement_process = psutil.Process()
        status = 'available'
    except ImportError:
        measurement_process = None
        status = 'not_collected: psutil unavailable'
    gpu = (
        shutil.which('nvidia-smi')
        or (
            '/usr/lib/wsl/lib/nvidia-smi'
            if Path('/usr/lib/wsl/lib/nvidia-smi').exists()
            else None
        )
    )
    docker = shutil.which('docker')
    volume_snapshot = {}
    if volume_name and docker and container_name:
        try:
            volume_snapshot['chroma_volume_size_bytes'] = int(
                subprocess.run(
                    [docker, 'exec', container_name, 'du', '-sb', '/data'],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                ).stdout.split(
                )[0],
            )
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
            volume_snapshot['chroma_volume_size_status'] = f'not_collected: {exc}'
    elif volume_path:
        try:
            volume_snapshot['chroma_volume_size_bytes'] = int(
                subprocess.run(
                    ['du', '-sb', volume_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                ).stdout.split(
                )[0],
            )
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
            volume_snapshot['chroma_volume_size_status'] = f'not_collected: {exc}'
    else:
        volume_snapshot['chroma_volume_size_status'] = (
            'not_collected: resources.chroma_volume_name/path not configured'
        )
    with path.open('w', encoding='utf-8') as stream:
        stream.write(
            json.dumps(
                {
                    'status': status,
                    'process_scope': (
                        'Python process and descendants; '
                        'configured vLLM PID tree when available'
                    ),
                    'vllm_pids': vllm_pids,
                    'docker_scope': (
                        f'configured Chroma container only: {container_name}'
                        if container_name
                        else 'not_collected: resources.chroma_container not configured'
                    ),
                    'gpu_scope': 'whole device; nvidia-smi',
                    'period_seconds_requested': interval,
                    'docker_period_seconds_requested': docker_interval,
                    'chroma_volume_scope': (
                        'single startup snapshot'
                        if volume_name or volume_path
                        else 'not_collected: volume name/path not configured'
                    ),
                },
            ) + '\n',
        )
        while not stop.is_set():
            sample_start = time.monotonic()
            record = {
                'sample_start_monotonic': sample_start,
                'sample_interval_actual_seconds': (
                    None if previous_start is None else sample_start - previous_start
                ),
            }
            previous_start = sample_start
            record.update(volume_snapshot)
            if measurement_process:
                try:
                    children = measurement_process.children(recursive=True)
                    record['python_process_tree'] = [
                        {'pid': p.pid, 'rss_bytes': p.memory_info().rss}
                        for p in [measurement_process, *children]
                    ]
                    vllm = []
                    for pid in [vllm_pids] if isinstance(vllm_pids, int) else vllm_pids:
                        try:
                            process = psutil.Process(int(pid))
                            vllm.extend(
                                [
                                    {'pid': p.pid, 'rss_bytes': p.memory_info().rss}
                                    for p in [process, *process.children(recursive=True)]
                                ],
                            )
                        except (psutil.Error, ValueError, OSError) as exc:
                            vllm.append({'pid': pid, 'status': f'not_collected: {exc}'})
                    record['vllm_process_tree'] = (
                        vllm if vllm else 'not_collected: resources.vllm_pids not configured'
                    )
                    memory = psutil.virtual_memory()
                    swap_memory = psutil.swap_memory()
                    record['wsl_memory_swap'] = {
                        'total_bytes': memory.total,
                        'available_bytes': memory.available,
                        'used_bytes': memory.used,
                        'swap_total_bytes': swap_memory.total,
                        'swap_used_bytes': swap_memory.used,
                    }
                    disk = psutil.disk_io_counters()
                    current = {
                        'read_bytes': disk.read_bytes,
                        'write_bytes': disk.write_bytes,
                        'read_count': disk.read_count,
                        'write_count': disk.write_count,
                    }
                    record['disk_io_cumulative'] = current
                    record['disk_io_delta'] = (
                        None
                        if previous_disk_io is None
                        else {k: current[k] - previous_disk_io[k] for k in current}
                    )
                    previous_disk_io = current
                except (psutil.Error, OSError) as exc:
                    record['psutil_error'] = str(exc)
            else:
                record['python_process_tree_status'] = 'not_collected: psutil unavailable'
                record['vllm_process_tree_status'] = 'not_collected: psutil unavailable'
                record['wsl_memory_swap_status'] = 'not_collected: psutil unavailable'
                record['disk_io_status'] = 'not_collected: psutil unavailable'
            if gpu:
                try:
                    record['nvidia_smi_device_wide'] = subprocess.run(
                        [
                            gpu,
                            '--query-gpu=memory.total,memory.used,utilization.gpu',
                            '--format=csv,noheader,nounits',
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=True,
                    ).stdout.strip(
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    record['nvidia_smi_status'] = f'not_collected: {exc}'
            else:
                record['nvidia_smi_status'] = 'not_collected: executable unavailable'
            now = time.monotonic()
            # Docker 子进程采用较慢周期，主机采样保留自己的实际间隔。
            if now >= next_docker:
                if docker and container_name:
                    try:
                        record['chroma_docker_stats'] = subprocess.run(
                            [docker, 'stats', '--no-stream', '--format', '{{json .}}', container_name],
                            capture_output=True,
                            text=True,
                            timeout=8,
                            check=True,
                        ).stdout.strip(
                        )
                    except (OSError, subprocess.SubprocessError) as exc:
                        record['chroma_docker_stats_status'] = f'not_collected: {exc}'
                else:
                    record['chroma_docker_stats_status'] = (
                        'not_collected: docker executable or '
                        'resources.chroma_container unavailable'
                    )
                next_docker = now + docker_interval
            record['sample_end_monotonic'] = time.monotonic()
            record['sample_duration_seconds'] = record['sample_end_monotonic'] - sample_start
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            stop.wait(max(0, interval - (time.monotonic() - sample_start)))
