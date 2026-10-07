/* Installation is progressive enhancement; research also works in a browser. */
(() => {
  let pendingInstall = null;
  const standalone = () => window.matchMedia('(display-mode: standalone)').matches || navigator.standalone;
  const sync = () => {
    const button = document.getElementById('install-app-button');
    if (button) button.hidden = !pendingInstall || standalone();
    const state = document.getElementById('installation-state');
    if (state) state.textContent = standalone() ? 'Installed on this device' : 'One app for your phone and computer';
    const banner = document.getElementById('connection-banner');
    if (banner) banner.hidden = navigator.onLine;
  };
  window.addEventListener('beforeinstallprompt', event => {
    event.preventDefault(); pendingInstall = event; sync();
  });
  window.addEventListener('appinstalled', () => { pendingInstall = null; sync(); });
  window.addEventListener('online', sync);
  window.addEventListener('offline', sync);
  document.addEventListener('click', async event => {
    if (!event.target.closest('#install-app-button') || !pendingInstall) return;
    const prompt = pendingInstall;
    pendingInstall = null; sync();
    try { await prompt.prompt(); await prompt.userChoice; }
    catch (_) { /* The manual installation steps remain available. */ }
  });
  // Dash mounts after this asset runs. Observe mounts without changing React state.
  const observer = new MutationObserver(() => {
    if (document.getElementById('installation-state')) { sync(); observer.disconnect(); }
  });
  observer.observe(document.documentElement, {childList:true, subtree:true});
  if ('serviceWorker' in navigator && window.isSecureContext) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('/service-worker.js', {scope:'/'}).catch(() => {
        // Installation can be unavailable under browser or private-mode policies.
      });
    });
  }
})();
