/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  output: 'standalone',
  // `next build` and `next dev` both write to the build directory, so building
  // while the dev server is running corrupts it (pages lose their CSS and JS
  // until dev is restarted). To check a production build with dev running:
  //   NEXT_DIST_DIR=.next-check npx next build
  distDir: process.env.NEXT_DIST_DIR || '.next',
};

module.exports = nextConfig;
