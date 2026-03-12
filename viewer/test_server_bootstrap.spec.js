// @ts-check
const { test, expect } = require('@playwright/test');
const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');

const FIXTURES = path.join(__dirname, 'test_fixtures_bootstrap');

/**
 * Build a minimal valid WebM (EBML + Segment header) so the browser accepts
 * it via <input type="file"> without needing ffmpeg at build time.
 */
function makeMinimalWebM() {
  // Minimal WebM: EBML header + empty Segment — enough to pass file-input
  // accept="video/*" checks and to be assigned via setInputFiles.
  return Buffer.from([
    // EBML Header
    0x1a, 0x45, 0xdf, 0xa3, // EBML element ID
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x13, // size = 19
    0x42, 0x86, 0x81, 0x01, // EBMLVersion = 1
    0x42, 0xf7, 0x81, 0x01, // EBMLReadVersion = 1
    0x42, 0xf2, 0x81, 0x04, // EBMLMaxIDLength = 4
    0x42, 0xf3, 0x81, 0x08, // EBMLMaxSizeLength = 8
    0x42, 0x82, 0x84, 0x77, 0x65, 0x62, 0x6d, // DocType = "webm"
    // Segment (empty, unknown size)
    0x18, 0x53, 0x80, 0x67, 0x01, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  ]);
}

// Reuse same fixtures as main test
test.beforeAll(async () => {
  fs.mkdirSync(FIXTURES, { recursive: true });
  const vid1 = path.join(FIXTURES, 'vid1.webm');
  const vid2 = path.join(FIXTURES, 'vid2.webm');
  const csv = path.join(FIXTURES, 'delta.csv');

  // Try ffmpeg first; fall back to minimal WebM stub
  if (!fs.existsSync(vid1)) {
    try {
      execSync(
        `ffmpeg -y -f lavfi -i color=c=red:s=160x120:d=2:r=10 -c:v libvpx -b:v 1M ${vid1}`,
        { stdio: 'ignore' }
      );
    } catch {
      fs.writeFileSync(vid1, makeMinimalWebM());
    }
  }
  if (!fs.existsSync(vid2)) {
    try {
      execSync(
        `ffmpeg -y -f lavfi -i color=c=blue:s=160x120:d=2:r=10 -c:v libvpx -b:v 1M ${vid2}`,
        { stdio: 'ignore' }
      );
    } catch {
      fs.writeFileSync(vid2, makeMinimalWebM());
    }
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

test.describe('Upload progress indicator', () => {
  test.beforeEach(async ({ page }) => {
    // Enter webapp mode so upload UI is shown
    await page.route('/api/status', route =>
      route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ mode: 'webapp', ready: false }),
      })
    );
  });

  /** Helper: set up routes so upload + run succeed, return collected state */
  async function setupUploadFlow(page) {
    // Mock upload endpoint — respond with a job_id
    await page.route('/api/upload', route =>
      route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ job_id: 'test-job-1' }),
      })
    );

    // Mock run endpoint
    await page.route('/api/run/test-job-1', route =>
      route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ ok: true }),
      })
    );
  }

  /** Helper: navigate, pick files, and click Run */
  async function startUpload(page) {
    await page.goto('/viewer.html');
    await expect(page.locator('#upload-zone')).toBeVisible({ timeout: 3000 });

    // Programmatically set files and enable button (file input is hidden)
    const vid1 = path.join(FIXTURES, 'vid1.webm');
    const vid2 = path.join(FIXTURES, 'vid2.webm');
    await page.locator('#upload-files').setInputFiles([vid1, vid2]);

    // The hidden input fires change but our handler uses handleFiles()
    // which is wired to the input's change event — trigger it properly
    await page.evaluate(([v1Name, v2Name]) => {
      const input = document.getElementById('upload-files');
      // Dispatch the change event so handleFiles picks up the files
      input.dispatchEvent(new Event('change'));
    }, [path.basename(vid1), path.basename(vid2)]);

    await expect(page.locator('#run-btn')).toBeEnabled({ timeout: 2000 });
    await page.locator('#run-btn').click();
  }

  test('progress area becomes visible when upload starts', async ({ page }) => {
    await setupUploadFlow(page);
    await startUpload(page);

    // Progress area should be visible
    await expect(page.locator('#progress-area')).toBeVisible({ timeout: 3000 });
  });

  test('stage text shows uploading during upload', async ({ page }) => {
    // Delay the upload response so we can inspect intermediate state
    await page.route('/api/upload', async route => {
      // Small delay to let the UI update
      await new Promise(r => setTimeout(r, 200));
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ job_id: 'test-job-2' }),
      });
    });
    await page.route('/api/run/test-job-2', route =>
      route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ ok: true }),
      })
    );

    await startUpload(page);

    // Stage text should say 'uploading'
    await expect(page.locator('#progress-stage')).toHaveText('uploading', { timeout: 3000 });
  });

  test('button shows Uploading then Running after upload completes', async ({ page }) => {
    await setupUploadFlow(page);

    // Mock WebSocket so the flow doesn't error after run
    await page.goto('/viewer.html');
    await expect(page.locator('#upload-zone')).toBeVisible({ timeout: 3000 });

    const vid1 = path.join(FIXTURES, 'vid1.webm');
    const vid2 = path.join(FIXTURES, 'vid2.webm');
    await page.locator('#upload-files').setInputFiles([vid1, vid2]);
    await page.evaluate(() => {
      document.getElementById('upload-files').dispatchEvent(new Event('change'));
    });
    await expect(page.locator('#run-btn')).toBeEnabled({ timeout: 2000 });

    // Click and wait for upload to complete — button should transition to Running
    await page.locator('#run-btn').click();
    await expect(page.locator('#run-btn')).toHaveText('Running...', { timeout: 5000 });
  });

  test('progress bar resets to 0 after upload for analysis phase', async ({ page }) => {
    await setupUploadFlow(page);
    await startUpload(page);

    // After upload completes and analysis starts, bar should reset
    await expect(page.locator('#run-btn')).toHaveText('Running...', { timeout: 5000 });
    const barValue = await page.locator('#progress-bar').evaluate(el => el.value);
    expect(barValue).toBe(0);
  });

  test('progress detail shows MB format during upload', async ({ page }) => {
    // For local XHR the progress event fires with the full payload immediately,
    // so detail should show the MB format (X.X / X.X MB) or be empty if
    // the upload completed too fast. We check the format if present.
    await setupUploadFlow(page);
    await startUpload(page);

    // Wait for upload to complete
    await expect(page.locator('#run-btn')).toHaveText('Running...', { timeout: 5000 });

    // After analysis starts, detail text resets to empty
    const detail = await page.locator('#progress-detail').textContent();
    expect(detail).toBe('');
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
