/* Shared helpers for AEGIS pages */
(function () {
  if (window.Chart) {
    Chart.defaults.color = "#8fa1c0";
    Chart.defaults.borderColor = "rgba(122,148,196,0.12)";
    Chart.defaults.font.family =
      "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    Chart.defaults.font.size = 11;
  }

  window.AEGIS = {
    accent: "#5b8dff",
    accent2: "#38d9c4",
    credible: "#2fcf8f",
    misleading: "#ff5d73",
    warn: "#ffb454",

    gradient(ctx, area, from, to) {
      if (!area) return from;
      const g = ctx.createLinearGradient(0, area.top, 0, area.bottom);
      g.addColorStop(0, from);
      g.addColorStop(1, to);
      return g;
    },

    formatPct(value) {
      return (Number(value) * 100).toFixed(1) + "%";
    },
  };

  document.addEventListener("DOMContentLoaded", function () {
    // Progress bars declared with data-fill="0.72"
    document.querySelectorAll("[data-fill]").forEach(function (el) {
      const pct = Math.max(0, Math.min(100, parseFloat(el.dataset.fill) || 0));
      el.style.width = pct + "%";
    });
  });
})();
