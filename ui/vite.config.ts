import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const API_TARGET = process.env.VITE_BACKEND_URL ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/planner': API_TARGET,
      '/extensions': API_TARGET,
    },
  },
  build: {
    outDir: 'dist',
  },
})
