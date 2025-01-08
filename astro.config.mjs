import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://yourusername.github.io',
  base: '', // Remove the base path to serve from root
  output: 'static'
});