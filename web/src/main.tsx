/** MemMgr 八页应用入口：绿色侧栏、响应式导航及真实业务页面。 */
import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BrainCircuit,
  ChartNoAxesCombined,
  Database,
  FileChartColumn,
  FlaskConical,
  Layers3,
  Menu,
  MessagesSquare,
  Settings2,
  Terminal,
  X,
} from "lucide-react";
import { Overview, Tasks } from "./operations";
import { QuestionAnswer, Logs } from "./streams";
import { DatabasePage } from "./database";
import { Configuration } from "./configuration";
import { Reports } from "./reports";
import "./styles.css";
const pages = [
  {
    id: "overview",
    title: "总览",
    icon: ChartNoAxesCombined,
  },
  {
    id: "qa",
    title: "交互式问答",
    icon: MessagesSquare,
  },
  {
    id: "tasks",
    title: "实验任务",
    icon: FlaskConical,
  },
  {
    id: "database",
    title: "数据库",
    icon: Database,
  },
  {
    id: "data",
    title: "数据与索引",
    icon: Layers3,
  },
  {
    id: "logs",
    title: "日志",
    icon: Terminal,
  },
  {
    id: "reports",
    title: "实验报告",
    icon: FileChartColumn,
  },
  {
    id: "config",
    title: "配置",
    icon: Settings2,
  },
];
/** 应用壳；hash 导航保留页面链接，移动端可折叠侧栏。 */
function App() {
  const [current, setCurrent] = useState(
    location.hash.slice(1).split("?")[0] || "overview",
  );
  const [open, setOpen] = useState(false);
  React.useEffect(() => {
    /** 读取浏览器 hash 导航。 */ function onHash() {
      setCurrent(location.hash.slice(1).split("?")[0] || "overview");
      setOpen(false);
    }
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const page = pages.find((p) => p.id === current) ?? pages[0];
  return (
    <div className="app">
      <aside className={open ? "sidebar open" : "sidebar"}>
        <a className="brand" href="#overview">
          <span className="brand-mark">
            <BrainCircuit size={25} />
          </span>
          <span>MemMgr</span>
        </a>
        <nav aria-label="主导航">
          {pages.map((p) => (
            <a
              key={p.id}
              href={`#${p.id}`}
              className={page.id === p.id ? "nav-item active" : "nav-item"}
              aria-current={page.id === p.id ? "page" : undefined}
            >
              <p.icon size={19} />
              <span>{p.title}</span>
            </a>
          ))}
        </nav>
      </aside>
      {open && (
        <button
          className="scrim"
          aria-label="关闭导航遮罩"
          onClick={() => setOpen(false)}
        />
      )}
      <div className="main-shell">
        <header className="topbar">
          <div className="topbar-title">
            <button
              className="menu-button secondary"
              aria-label={open ? "关闭导航" : "展开导航"}
              onClick={() => setOpen(!open)}
            >
              {open ? <X size={21} /> : <Menu size={21} />}
            </button>
            <span>
              工作空间 <span className="slash">/</span>{" "}
              <strong>{page.title}</strong>
            </span>
          </div>
          <a className="topbar-link" href="#config">
            <Settings2 size={16} />
            运行配置
          </a>
        </header>
        <main>
          <div className="page-heading">
            <h1>{page.title}</h1>
          </div>
          {page.id === "overview" ? (
            <Overview
              navigate={(id) => {
                location.hash = id;
              }}
            />
          ) : page.id === "qa" ? (
            <QuestionAnswer />
          ) : page.id === "tasks" ? (
            <Tasks key="tasks" />
          ) : page.id === "data" ? (
            <Tasks key="data" dataMode />
          ) : page.id === "database" ? (
            <DatabasePage />
          ) : page.id === "logs" ? (
            <Logs />
          ) : page.id === "reports" ? (
            <Reports />
          ) : (
            <Configuration />
          )}
        </main>
      </div>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
