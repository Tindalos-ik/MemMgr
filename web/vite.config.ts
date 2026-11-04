/** MemMgr 前端开发服务器：将 API 请求转发到本机 FastAPI。 */
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  server: { host: "127.0.0.1", proxy: { "/api": "http://127.0.0.1:8765" } },
});
