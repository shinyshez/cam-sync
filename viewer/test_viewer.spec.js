// @ts-check
const { test, expect } = require('@playwright/test');
const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');

const FIXTURES = path.join(__dirname, 'test_fixtures');

// Generate tiny test videos + CSV before all tests
test.beforeAll(async () => {
  fs.mkdirSync(FIXTURES, { recursive: true });

  const vid1 = path.join(FIXTURES, 'vid1.webm');
  const vid2 = path.join(FIXTURES, 'vid2.webm');
  const csv = path.join(FIXTURES, 'delta.csv');

  // 2-second solid color videos at 10fps, VP8/WebM for Chromium codec support
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

  // Simple delta CSV: vid2 time -> delta (vid1 is ahead by varying amounts)
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

// ── Page loads ──────────────────────────────────────────────────────────

test.describe('Page loads', () => {
  test('renders setup panel with file inputs', async ({ page }) => {
    await page.goto('/viewer.html');
    await expect(page.locator('#setup-panel')).toBeVisible();
    await expect(page.locator('input[type="file"]')).toHaveCount(3);
  });

  test('player panel is hidden initially', async ({ page }) => {
    await page.goto('/viewer.html');
    await expect(page.locator('#player-panel')).toBeHidden();
  });
});

// ── CSV parsing unit tests ──────────────────────────────────────────────

test.describe('CSV parsing (parseDeltaCSV)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/viewer.html');
  });

  test('parses header + data rows correctly', async ({ page }) => {
    const result = await page.evaluate(() => {
      const csv = 'time_vid2_seconds,delta_seconds\n0.0,2.0\n5.0,3.0\n10.0,2.5';
      const { times, deltas } = window.parseDeltaCSV(csv);
      return { times: Array.from(times), deltas: Array.from(deltas) };
    });
    expect(result.times).toEqual([0.0, 5.0, 10.0]);
    expect(result.deltas).toEqual([2.0, 3.0, 2.5]);
  });

  test('deduplicates timestamps (last value wins)', async ({ page }) => {
    const result = await page.evaluate(() => {
      const csv = 'time_vid2_seconds,delta_seconds\n0.0,1.0\n5.0,2.0\n5.0,3.0\n10.0,4.0';
      const { times, deltas } = window.parseDeltaCSV(csv);
      return { times: Array.from(times), deltas: Array.from(deltas) };
    });
    expect(result.times).toEqual([0.0, 5.0, 10.0]);
    expect(result.deltas).toEqual([1.0, 3.0, 4.0]);
  });

  test('returns Float64Array pair', async ({ page }) => {
    const result = await page.evaluate(() => {
      const csv = 'time_vid2_seconds,delta_seconds\n0.0,1.0';
      const { times, deltas } = window.parseDeltaCSV(csv);
      return {
        timesType: times.constructor.name,
        deltasType: deltas.constructor.name,
      };
    });
    expect(result.timesType).toBe('Float64Array');
    expect(result.deltasType).toBe('Float64Array');
  });

  test('handles Windows-style CRLF line endings', async ({ page }) => {
    const result = await page.evaluate(() => {
      const csv = 'time_vid2_seconds,delta_seconds\r\n0.0,2.0\r\n5.0,3.0';
      const { times, deltas } = window.parseDeltaCSV(csv);
      return { times: Array.from(times), deltas: Array.from(deltas) };
    });
    expect(result.times).toEqual([0.0, 5.0]);
    expect(result.deltas).toEqual([2.0, 3.0]);
  });
});

// ── Interpolation tests ─────────────────────────────────────────────────

test.describe('Interpolation (interpDelta)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/viewer.html');
  });

  test('exact sample points return correct delta', async ({ page }) => {
    const result = await page.evaluate(() => {
      const times = new Float64Array([0, 5, 10]);
      const deltas = new Float64Array([2.0, 3.0, 2.5]);
      return [
        window.interpDelta(0, times, deltas),
        window.interpDelta(5, times, deltas),
        window.interpDelta(10, times, deltas),
      ];
    });
    expect(result).toEqual([2.0, 3.0, 2.5]);
  });

  test('midpoint interpolation', async ({ page }) => {
    const result = await page.evaluate(() => {
      const times = new Float64Array([0, 10]);
      const deltas = new Float64Array([2.0, 4.0]);
      return window.interpDelta(5, times, deltas);
    });
    expect(result).toBeCloseTo(3.0, 5);
  });

  test('clamps before start (returns first delta)', async ({ page }) => {
    const result = await page.evaluate(() => {
      const times = new Float64Array([5, 10]);
      const deltas = new Float64Array([2.0, 3.0]);
      return window.interpDelta(0, times, deltas);
    });
    expect(result).toBe(2.0);
  });

  test('clamps after end (returns last delta)', async ({ page }) => {
    const result = await page.evaluate(() => {
      const times = new Float64Array([0, 5]);
      const deltas = new Float64Array([2.0, 3.0]);
      return window.interpDelta(99, times, deltas);
    });
    expect(result).toBe(3.0);
  });

  test('quarter-point interpolation', async ({ page }) => {
    const result = await page.evaluate(() => {
      const times = new Float64Array([0, 10]);
      const deltas = new Float64Array([0, 4.0]);
      return window.interpDelta(2.5, times, deltas);
    });
    expect(result).toBeCloseTo(1.0, 5);
  });
});

// ── File loading & player visibility ────────────────────────────────────

test.describe('File loading', () => {
  test('loading all three files shows player panel', async ({ page }) => {
    await page.goto('/viewer.html');

    // Load files via file inputs
    const [vid1Input, vid2Input, csvInput] = await page.locator('input[type="file"]').all();
    await vid1Input.setInputFiles(path.join(FIXTURES, 'vid1.webm'));
    await vid2Input.setInputFiles(path.join(FIXTURES, 'vid2.webm'));
    await csvInput.setInputFiles(path.join(FIXTURES, 'delta.csv'));

    // Player panel should appear
    await expect(page.locator('#player-panel')).toBeVisible({ timeout: 3000 });
    await expect(page.locator('#setup-panel')).toBeHidden();
  });
});

// ── Layout ──────────────────────────────────────────────────────────────

test.describe('Layout', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/viewer.html');
    const [vid1Input, vid2Input, csvInput] = await page.locator('input[type="file"]').all();
    await vid1Input.setInputFiles(path.join(FIXTURES, 'vid1.webm'));
    await vid2Input.setInputFiles(path.join(FIXTURES, 'vid2.webm'));
    await csvInput.setInputFiles(path.join(FIXTURES, 'delta.csv'));
    await expect(page.locator('#player-panel')).toBeVisible({ timeout: 3000 });
  });

  test('two video elements visible, stacked vertically', async ({ page }) => {
    const vid1 = page.locator('#vid1');
    const vid2 = page.locator('#vid2');
    await expect(vid1).toBeVisible();
    await expect(vid2).toBeVisible();

    // vid1 should be above vid2
    const box1 = await vid1.boundingBox();
    const box2 = await vid2.boundingBox();
    expect(box1.y).toBeLessThan(box2.y);
  });
});

// ── Delta overlay ───────────────────────────────────────────────────────

test.describe('Delta overlay', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/viewer.html');
    const [vid1Input, vid2Input, csvInput] = await page.locator('input[type="file"]').all();
    await vid1Input.setInputFiles(path.join(FIXTURES, 'vid1.webm'));
    await vid2Input.setInputFiles(path.join(FIXTURES, 'vid2.webm'));
    await csvInput.setInputFiles(path.join(FIXTURES, 'delta.csv'));
    await expect(page.locator('#player-panel')).toBeVisible({ timeout: 3000 });
  });

  test('delta overlay is visible', async ({ page }) => {
    await expect(page.locator('#delta-overlay')).toBeVisible();
  });

  test('delta overlay shows a numeric value', async ({ page }) => {
    const text = await page.locator('#delta-overlay').textContent();
    // Should contain a number like +0.50s or -0.50s
    expect(text).toMatch(/[+-]?\d+\.\d+s/);
  });
});

// ── Delta graph ─────────────────────────────────────────────────────────

test.describe('Delta graph', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/viewer.html');
    const [vid1Input, vid2Input, csvInput] = await page.locator('input[type="file"]').all();
    await vid1Input.setInputFiles(path.join(FIXTURES, 'vid1.webm'));
    await vid2Input.setInputFiles(path.join(FIXTURES, 'vid2.webm'));
    await csvInput.setInputFiles(path.join(FIXTURES, 'delta.csv'));
    await expect(page.locator('#player-panel')).toBeVisible({ timeout: 3000 });
    // Allow canvas to render (headless Chromium can be slower)
    await page.waitForTimeout(200);
  });

  test('canvas element is visible', async ({ page }) => {
    await expect(page.locator('#delta-graph')).toBeVisible();
  });

  test('canvas has nonzero dimensions', async ({ page }) => {
    const box = await page.locator('#delta-graph').boundingBox();
    expect(box.width).toBeGreaterThan(50);
    expect(box.height).toBeGreaterThan(20);
  });

  test('canvas has drawn content (not blank)', async ({ page }) => {
    // Check that the canvas isn't entirely transparent/empty
    const hasContent = await page.evaluate(() => {
      const canvas = document.getElementById('delta-graph');
      const ctx = canvas.getContext('2d');
      const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      // Check if any pixel has non-zero alpha
      for (let i = 3; i < data.length; i += 4) {
        if (data[i] > 0) return true;
      }
      return false;
    });
    expect(hasContent).toBe(true);
  });

  test('playhead line is present', async ({ page }) => {
    // The playhead is drawn at x=0 when time=0. Check for a vertical stripe
    // of bright pixels near the left edge of the canvas.
    const hasPlayhead = await page.evaluate(() => {
      const canvas = document.getElementById('delta-graph');
      const ctx = canvas.getContext('2d');
      // Sample a narrow column near x=1 (playhead at t=0)
      const data = ctx.getImageData(0, 0, 3, canvas.height).data;
      let brightPixels = 0;
      for (let i = 0; i < data.length; i += 4) {
        // Check for white-ish playhead pixels (R+G+B > 500)
        if (data[i] + data[i+1] + data[i+2] > 500 && data[i+3] > 200) brightPixels++;
      }
      return brightPixels > 3;
    });
    expect(hasPlayhead).toBe(true);
  });
});

// ── Controls ────────────────────────────────────────────────────────────

test.describe('Controls', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/viewer.html');
    const [vid1Input, vid2Input, csvInput] = await page.locator('input[type="file"]').all();
    await vid1Input.setInputFiles(path.join(FIXTURES, 'vid1.webm'));
    await vid2Input.setInputFiles(path.join(FIXTURES, 'vid2.webm'));
    await csvInput.setInputFiles(path.join(FIXTURES, 'delta.csv'));
    await expect(page.locator('#player-panel')).toBeVisible({ timeout: 3000 });
  });

  test('play/pause button present', async ({ page }) => {
    await expect(page.locator('#play-btn')).toBeVisible();
  });

  test('scrub bar present', async ({ page }) => {
    await expect(page.locator('#scrub-bar')).toBeVisible();
  });

  test('speed selector present with options', async ({ page }) => {
    const speed = page.locator('#speed-select');
    await expect(speed).toBeVisible();
    const options = await speed.locator('option').allTextContents();
    expect(options).toContain('1x');
  });

  test('time display present', async ({ page }) => {
    await expect(page.locator('#time-display')).toBeVisible();
  });

  test('play button toggles to pause on click', async ({ page }) => {
    const btn = page.locator('#play-btn');
    const textBefore = await btn.textContent();
    await btn.click();
    // Give a moment for state change
    await page.waitForTimeout(100);
    const textAfter = await btn.textContent();
    expect(textBefore).not.toBe(textAfter);
  });

  test('space key toggles play/pause', async ({ page }) => {
    const btn = page.locator('#play-btn');
    const textBefore = await btn.textContent();
    await page.keyboard.press('Space');
    await page.waitForTimeout(100);
    const textAfter = await btn.textContent();
    expect(textBefore).not.toBe(textAfter);
  });
});

// ── Editable labels ─────────────────────────────────────────────────────

test.describe('Editable labels', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/viewer.html');
    const [vid1Input, vid2Input, csvInput] = await page.locator('input[type="file"]').all();
    await vid1Input.setInputFiles(path.join(FIXTURES, 'vid1.webm'));
    await vid2Input.setInputFiles(path.join(FIXTURES, 'vid2.webm'));
    await csvInput.setInputFiles(path.join(FIXTURES, 'delta.csv'));
    await expect(page.locator('#player-panel')).toBeVisible({ timeout: 3000 });
  });

  test('rider labels are visible and editable', async ({ page }) => {
    const label1 = page.locator('#label-vid1');
    const label2 = page.locator('#label-vid2');
    await expect(label1).toBeVisible();
    await expect(label2).toBeVisible();
    // contenteditable
    expect(await label1.getAttribute('contenteditable')).toBe('true');
    expect(await label2.getAttribute('contenteditable')).toBe('true');
  });
});
