// Clicking the toolbar icon opens the side panel for the current tab. The
// click also grants tab-capture access to that tab, which the panel uses
// when you press Start.

chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: false }).catch(() => {});

chrome.action.onClicked.addListener((tab) => {
  // Must be called synchronously inside the user gesture.
  chrome.sidePanel.open({ windowId: tab.windowId });
  chrome.storage.session.set({ invokedTabId: tab.id });
  chrome.runtime.sendMessage({ type: 'invoked', tabId: tab.id }).catch(() => {});
});
