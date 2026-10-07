# MarketSignal on iPhone, Windows and Mac

MarketSignal is an installable browser app (PWA). It opens in its own window and
uses the same hosted service on all devices. It is not an App Store binary.
Research calculations and fresh data require an internet connection.

## Install from the hosted HTTPS address

- **iPhone / iPad:** open the address in Safari, tap Share, then **Add to Home
  Screen**. Turn on **Open as Web App** if offered. Launch the MarketSignal icon.
- **Windows:** open in Edge or Chrome and use **Install app** in MarketSignal
  when offered, or the browser's address-bar install icon/menu. Pin it to Start
  or the taskbar if desired.
- **Mac:** install through Chrome/Edge, or on supported macOS versions use
  Safari → File → **Add to Dock**.

Installation prompts depend on the browser. If no automatic button appears, use
these manual steps. Browser access remains available without installing.

## What is saved

The chosen ticker and workspace are remembered on each device. Pending research
and study IDs survive a reload within the same browser session. Data snapshots,
reports, protocols and final study results live on the server's persistent disk.
They are not copied to the phone and are not private account data. There is no
account login or automatic per-user history synchronization between devices yet.

The Studies screen can resume the existing study for the same exact dataset.
A refreshed dataset can have a new identity. Download study/report records to
keep their identifiers and full details. Do not clear server storage to restart
an interrupted final evaluation.

## Offline and updates

Only a public offline landing page and icons are cached by the service worker.
Prices, research dashboards, downloads and API responses are never added to its
cache. Reconnect to load the app or refresh results. A job already submitted may
continue on the server when the phone disconnects; inspect its status on return.
A screen already open at disconnection may still show its previous results and
an offline notice, not live prices.

App updates are loaded from the server. Close all app/browser windows and reopen
to allow a waiting service worker update to activate. Updates do not force a
reload during research. The service worker is served at the origin root; deploy
this application at `/`, not under a subpath.

## Hosting needed for use across devices

An HTTPS deployment is required for installation on a phone. A running cloud
coding workspace is not a durable user-facing deployment. The checked-in Render
blueprint specifies a single paid web service with a persistent 1 GB disk and
`MARKETSIGNAL_DATA_DIR=/var/data/marketsignal`. Review the provider's current price
before creating it. No paid service is created by editing this file.

1. Put this tested revision on the GitHub branch selected for deployment.
2. In Render, create a Blueprint from that repository/branch and review its service
   and disk cost. The blueprint supplies the build command, start command,
   runtime version, data path and readiness probe.
3. Deploy and wait for `/ready` to return HTTP 200. Render supplies an HTTPS address.
4. Run `python scripts/smoke.py --base-url https://YOUR-SERVICE.onrender.com --research`
   from the pinned environment. Confirm report download, study workflow, and that
   the disk survives a service restart before relying on retention.
5. Open that same address on iPhone, Windows and Mac and install as above.

This is a shared research preview. Broader public distribution still needs the
release gates in [public-beta.md](public-beta.md), including provider rights,
access controls and production monitoring. Real-money trading is not available.

## Verification

Python tests check the manifest, real PNG dimensions, service-worker scope,
offline route and mounted workspaces. `scripts/app-browser-smoke.cjs` checks
installation metadata, service-worker activation, navigation, responsive layouts,
keyboard focus and offline recovery using Playwright/Chromium. Run it with a
separately installed Playwright (not an application runtime dependency):

```bash
MARKETSIGNAL_BASE_URL=http://127.0.0.1:8050 node scripts/app-browser-smoke.cjs
```

Localhost is for developer testing only. Emulation does not substitute for a
physical iPhone/Safari installation check on the final HTTPS deployment.
