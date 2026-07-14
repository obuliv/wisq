import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Local `npm run dev` proxy; in docker-compose, nginx.conf does this instead.
      "/api": "http://localhost:8000",
    },
  },
});
