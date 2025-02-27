/** @type {import('next').NextConfig} */
const nextConfig = {
    webpack: (config) => {
      config.module.rules.push({
        test: /@huggingface\/transformers/,
        type: "javascript/auto",
      });
  
      config.resolve.fallback = {
        ...config.resolve.fallback,
        path: require.resolve("path-browserify"),
        url: require.resolve("url/"),
      };
  
      return config;
    },
  };
  
  module.exports = nextConfig;