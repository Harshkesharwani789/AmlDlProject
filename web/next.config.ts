import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  poweredByHeader: false,
  compress: true,
  reactStrictMode: true,
  // Optimize server-side packages
  serverExternalPackages: ["fs-extra"], 
  images: {
    formats: ['image/avif', 'image/webp'],
    remotePatterns: [],
  },
  experimental: {
    // Enable any relevant experimental features for performance
    optimizePackageImports: ['lucide-react', 'framer-motion'],
  },
};

export default nextConfig;
