const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: '.',
  testMatch: ['test_viewer.spec.js', 'test_server_bootstrap.spec.js'],
  workers: 1,
  use: {
    baseURL: 'http://localhost:8787',
    launchOptions: {
      args: ['--no-sandbox'],
    },
  },
  webServer: {
    command: 'python3 -m http.server 8787',
    port: 8787,
    reuseExistingServer: true,
  },
});
