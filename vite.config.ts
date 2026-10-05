import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  base: '/Byte-Bot/',
  plugins: [react(), tailwindcss()],
  server: {
    // The trainer writes logs and checkpoints continuously; don't reload the page for them.
    watch: { ignored: ['**/training/runs/**'] },
  },
  build: {
    // Rapier's WebAssembly is inlined into the bundle (~2 MB).
    chunkSizeWarningLimit: 4000,
  },
})
