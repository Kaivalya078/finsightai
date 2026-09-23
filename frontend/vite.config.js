import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ command, mode }) => {
  // A build without a real API URL ships a frontend that silently calls
  // localhost. Set VITE_API_BASE in the Vercel project (or the shell).
  const env = loadEnv(mode, process.cwd())
  if (command === 'build' && !env.VITE_API_BASE?.startsWith('https://')) {
    throw new Error('Set VITE_API_BASE to the deployed API (https://...) before building')
  }
  return { plugins: [react()] }
})
