import { resolve } from 'node:path';

import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/postcss';
import { defineConfig } from 'vite';

export default defineConfig({
  base: '/spectrefaces/',
  root: resolve(import.meta.dirname, 'github-pages'),
  publicDir: resolve(import.meta.dirname, 'public'),
  resolve: {
    alias: {
      'next/image': resolve(import.meta.dirname, 'github-pages/next-image.tsx'),
      '@': import.meta.dirname,
    },
  },
  css: { postcss: { plugins: [tailwindcss()] } },
  plugins: [react()],
  build: {
    emptyOutDir: true,
    outDir: resolve(import.meta.dirname, 'dist-pages'),
  },
});
