// @ts-check
const { test, expect } = require('@playwright/test');
const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');

const FIXTURES = path.join(__dirname, 'test_fixtures_bootstrap');

// Reuse same fixtures as main test
test.beforeAll(async () => {
  fs.mkdirSync(FIXTURES, { recursive: true });
  const vid1 = path.join(FIXTURES, 'vid1.webm');
  const vid2 = path.join(FIXTURES, 'vid2.webm');
  const csv = path.join(FIXTURES, 'delta.csv');

  if (!fs.existsSync(vid1)) {
    execSync(
      `ffmpeg -y -f lavfi -i color=c=red:s=160x120:d=2:r=10 -c:v libvpx -b:v 1M ${vid1}`,
      { stdio: 'ignore' }
    );
  }
  if (!fs.existsSync(vid2)) {
    execSync(
      `ffmpeg -y -f lavfi -i color=c=blue:s=160x120:d=2:r=10 -c:v libvpx -b:v 1M ${vid2}`,
      { stdio: 'ignore' }
    );
  }
  if (!fs.existsSync(csv)) {
    fs.writeFileSync(csv, [
      'time_vid2_seconds,delta_seconds',
      '0.0,0.5',
      '1.0,1.0',
      '2.0,0.8',
    ].join('\n'));
  }
});

test.afterAll(async () => {
  fs.rmSync(FIXTURES, { recursive: true, force: true });
});

test.describe('Server bootstrap - no server (file mode)', () => {
  test('api/status 404 keeps file inputs visible (regression)', async ({ page }) => {
    // When served by python3 -m http.server, /api/status returns 404
    // The bootstrap should silently exit and file inputs remain
    await page.goto('/viewer.html');
    await expect(page.locator('#setup-panel')).toBeVisible();
    await expect(page.locator('#input-vid1')).toBeVisible();
    await expect(page.locator('#input-vid2')).toBeVisible();
    await expect(page.locator('#input-csv')).toBeVisible();
    // Upload UI should NOT be shown
    await expect(page.locator('#upload-zone')).toHaveCount(0);
  });

  test('file inputs still work after bootstrap (regression)', async ({ page }) => {
    await page.goto('/viewer.html');
    // Wait a moment for bootstrap to complete
    await page.waitForTimeout(200);

    const [vid1Input, vid2Input, csvInput] = await page.locator('input[type="file"]').all();
    await vid1Input.setInputFiles(path.join(FIXTURES, 'vid1.webm'));
    await vid2Input.setInputFiles(path.join(FIXTURES, 'vid2.webm'));
    await csvInput.setInputFiles(path.join(FIXTURES, 'delta.csv'));

    await expect(page.locator('#player-panel')).toBeVisible({ timeout: 3000 });
  });
});

test.describe('Server bootstrap - Mode D complete flow', () => {
  test('pipeline complete transitions to player (not stuck on upload UI)', async ({ page }) => {
    // Mock /api/status → webapp mode
    await page.route('/api/status', route =>
      route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ mode: 'webapp', ready: false }),
      })
    );

    await page.goto('/viewer.html');
    await expect(page.locator('#upload-zone')).toBeVisible({ timeout: 3000 });

    // Now simulate what happens after pipeline completes:
    // loadResultsFromServer() is called, which needs video + CSV endpoints
    const vid1Data = fs.readFileSync(path.join(FIXTURES, 'vid1.webm'));
    const vid2Data = fs.readFileSync(path.join(FIXTURES, 'vid2.webm'));
    await page.route('/api/video/1', route =>
      route.fulfill({ contentType: 'video/webm', body: vid1Data })
    );
    await page.route('/api/video/2', route =>
      route.fulfill({ contentType: 'video/webm', body: vid2Data })
    );
    const csvData = fs.readFileSync(path.join(FIXTURES, 'delta.csv'), 'utf-8');
    await page.route('/api/delta.csv', route =>
      route.fulfill({ contentType: 'text/csv', body: csvData })
    );

    // Call loadResultsFromServer() — this is what the WS complete handler does
    const error = await page.evaluate(() => {
      try {
        loadResultsFromServer();
        return null;
      } catch (e) {
        return e.message;
      }
    });
    expect(error).toBeNull();

    // Player should become visible
    await expect(page.locator('#player-panel')).toBeVisible({ timeout: 5000 });
  });
});

test.describe('Server bootstrap - mock server responses', () => {
  test('ready=true auto-activates player', async ({ page }) => {
    // Mock /api/status to return ready
    await page.route('/api/status', route =>
      route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ mode: 'cli', ready: true }),
      })
    );

    // Mock video endpoints — return a real video
    const vid1Data = fs.readFileSync(path.join(FIXTURES, 'vid1.webm'));
    const vid2Data = fs.readFileSync(path.join(FIXTURES, 'vid2.webm'));
    await page.route('/api/video/1', route =>
      route.fulfill({ contentType: 'video/webm', body: vid1Data })
    );
    await page.route('/api/video/2', route =>
      route.fulfill({ contentType: 'video/webm', body: vid2Data })
    );

    // Mock delta CSV
    const csvData = fs.readFileSync(path.join(FIXTURES, 'delta.csv'), 'utf-8');
    await page.route('/api/delta.csv', route =>
      route.fulfill({ contentType: 'text/csv', body: csvData })
    );

    await page.goto('/viewer.html');

    // Player panel should auto-activate
    await expect(page.locator('#player-panel')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('#setup-panel')).toBeHidden();
  });

  test('webapp mode shows upload UI', async ({ page }) => {
    await page.route('/api/status', route =>
      route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ mode: 'webapp', ready: false }),
      })
    );

    await page.goto('/viewer.html');

    // Upload UI should be visible
    await expect(page.locator('#upload-zone')).toBeVisible({ timeout: 3000 });
    // File inputs should be hidden (replaced by upload UI)
    await expect(page.locator('#input-vid1')).toBeHidden();
  });
});
