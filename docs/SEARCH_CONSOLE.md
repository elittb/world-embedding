# Google Search Console for this project

You **cannot** verify `https://github.com/elittb/world-embedding` in Search Console. GitHub owns that page; you cannot add Google’s HTML tag or upload a verification file at the URL Google requires.

You **can** verify **`https://elittb.github.io/world-embedding/`** (GitHub Pages). That site is under your control via the `docs/` folder in this repo.

## Step 1: Turn on GitHub Pages

1. Open the repo on GitHub: **Settings** → **Pages** (left sidebar).
2. Under **Build and deployment** → **Source**, choose **Deploy from a branch**.
3. **Branch:** `main`, folder **`/docs`**, then **Save**.
4. After 1–2 minutes, the site should load at:
   **`https://elittb.github.io/world-embedding/`**

(If the URL 404s, wait a few minutes and hard-refresh.)

## Step 2: Add the property in Search Console

1. Go to [Google Search Console](https://search.google.com/search-console).
2. **Add property** → **URL prefix**.
3. Enter exactly: `https://elittb.github.io/world-embedding/`
4. Choose **HTML tag** verification.
5. Copy the `<meta name="google-site-verification" content="..." />` line Google shows you.

## Step 3: Put the meta tag in `docs/index.html`

1. Open `docs/index.html` in this repo.
2. Paste the meta tag **inside** `<head>`, right after the comment that mentions Search Console (before `<style>`).
3. Commit and push to `main`. Wait for Pages to rebuild (~1 minute).
4. In Search Console, click **Verify**.

## Step 4: Ask Google to index the site

In Search Console, use **URL inspection** for `https://elittb.github.io/world-embedding/` and request indexing.

## Optional: link from README

Add a badge or line in the main README pointing to the Pages URL so users and crawlers find it. The GitHub repo URL will still be the primary home for stars and code; the Pages site helps **search** and verification.
