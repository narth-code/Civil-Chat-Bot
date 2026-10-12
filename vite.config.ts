import { defineConfig } from "vite";

const apiProxy = {
  "/api": {
    target: "http://127.0.0.1:8000",
    changeOrigin: true,
  },
};

export default defineConfig({
  publicDir: "assets",
  server: { host: "127.0.0.1", proxy: apiProxy },
  preview: { host: "127.0.0.1", proxy: apiProxy },
});
