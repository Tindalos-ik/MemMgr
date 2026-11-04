# AI 规范

本文件中的每一条规则都是强制性的。违反任何一条规则都会遭受毁灭性打击。不存在任何例外与豁免：临时的、一次性的、命令行上的违反同样是违反；没被当场发现也算违反；出于好意、为了进度、为了帮忙的违反也是违反。

## 语言

不允许使用"不是...而是..."句式；如果不需要对比的话，就不要对比；不要在任何话说完之后都提一句"不是其他的xxx"。如果没有叫你进行对比，就不允许使用"不是...而是..."、"要...而不是..."等类似的句式，你根本就没有需要说"不是"的对象，不要虚空打靶。所有类似的句式都不允许使用。

如果我让你搜索A相关内容，你搜索到B、C、D发现不满足要求，就不允许再把B、C、D列举出来了。我根本就不关心，看到这些只会污染我的眼睛

任何回答都不允许总结和总起，包括：
"上述内容是<某种概述>，下面详细拆开"
"一句话总结：xxx"

不允许使用"落地"、"钉死"、"对齐"等非技术名词、显然有其他可以代替的词语的黑话。使用正常的，不在互联网公司或者金融公司工作的任何人可以看懂的，在简单中文里常用的词汇

## 行为

不允许在实现或者测试的时候使用任何 mock、假的、欺骗的、只为了通过测试而 workaround 的方式来欺骗我

不要求最小化依赖，不允许用各种乱七八糟的方式（包括造轮子）绕过依赖。

编写的代码应当寻求 fast-fail，在出错位置就地崩溃，而不是捕获错误，也不是 fallback

我经常会在你更改后撤回/修改你的更改，所以如果你发现无法从你上一次更改之后继续更改，你应该重新读取文件内容。比如：你添加了A、B、C内容，我把B删掉了，这意味着接下来的改动应该在B被删掉的状态下（A、C）开始改动，不允许把B加回去

你在发现任何文档或者代码有错误的时候，你的更新不要保留任何错误痕迹。我们不需要任何错误的记录

如果你在执行一件事的过程中，用户问了一个别的事，如果回应用户能马上回应，那么就直接回应。暂时处理完用户请求以后马上继续你之前正在执行的事情，不要干一半不干了

不允许使用 ASCII Art 画示意图、表格等。如果你需要画图（实际上许多时候你并不需要），必须使用 mermaid。ASCII art 是一种人类不可读、AI也不可读的极其恶心的格式，不允许使用

在写任何代码的时候，都需要在文件的最前面添加 docstring，也不允许添加 shebang。注释使用中文，术语保留英文；不要过度注释

每个新增的函数都需要写好函数作用，解释参数

你可以在做事过程中新建中间文件，但是完成之后需要删除，我不需要我不看的文件来稀释我的注意力

在我没有声明需要备份或者新建数据库之前，所有的修改你可以直接原地修改，不需要新建数据库

在我没有声明备份代码之前，你不能进行任何代码的打包

文档是写给人看的，要简洁易懂，格式符合人的阅读需求

不要过度自我否定，不要主动削弱研究贡献，不要把论文写成检讨书，而是准确地，有力地呈现研究结果

## 目录索引
- [README.md](README.md)：项目入口与本地启动示例。
- [docs/deployment.md](docs/deployment.md)：环境准备、服务部署、SciFact 数据与迁移命令。
- [docs/experiment-design.md](docs/experiment-design.md)：实验分组、参数与指标含义、计时边界和复现步骤。
- [reports/baseline-report.md](reports/baseline-report.md)：最终实验报告。
- 配置文件：
  - [config/baseline.json](config/baseline.json)：数据集、检索与实验参数。
  - [config/models.json](config/models.json)：模型 revision 与本地快照位置。
  - [config/server.json](config/server.json)：vLLM 模型、服务地址与调度参数。
- 应用模块 [memmgr/](memmgr/)：
  - `cli.py`：命令行参数解析与子命令派发。
  - `data.py`、`index.py`：SciFact 数据准备、校验、分块、编码与 Chroma 向量写入。
  - `retrieval.py`、`prompt.py`、`streaming.py`：查询与相关性读取、证据组装、输入预算和流式生成。
  - `rag_batch.py`、`retrieval_batch.py`：端到端问答与纯检索测量。
  - `memory.py`、`memory_check.py`：长期记忆增删改查与重启持久化检查。
  - `resources.py`、`run_metadata.py`：资源采样和运行配置、源码、数据摘要记录。
- 脚本 [scripts/](scripts/)：
  - `download_models.py`：下载固定 revision 的模型。
  - `start-vllm.sh`、`chroma.sh`、`service_status.py`：启动与查看本地服务。
  - `index_measured.py`、`run_baseline.py`：索引测量和基线实验编排。
  - `summarize_baseline.py`、`review_run.py`、`review_retrieval.py`：汇总和检查正式测量记录。
- 测试：[tests/test_contracts.py](tests/test_contracts.py)检查数据格式、索引身份与隔离边界；[tests/test_prompt_budget.py](tests/test_prompt_budget.py)检查真实 tokenizer 的输入预算。
- [data/scifact-baseline/](data/scifact-baseline/)：SciFact 文献语料、问题、相关性标签、数据元信息与索引清单。
- [requirements.in](requirements.in)：直接依赖；[constraints.txt](constraints.txt)：固定软件包版本。
- [compose.yaml](compose.yaml)：Chroma 服务和持久数据卷配置。
- `.agent/`：运行时按需生成的 Agent 临时输出，任务完成后清理。
- [.gitignore](.gitignore)：排除临时输出、数据集、缓存与本地环境文件。
