const activity = document.querySelector("#activity");

for (const form of document.querySelectorAll("form[data-busy]")) {
  form.addEventListener("submit", () => {
    const button = form.querySelector("button[type='submit']");
    if (button) {
      button.disabled = true;
      button.textContent = form.dataset.busyButton || "Procesando…";
    }
    if (activity) {
      activity.hidden = false;
      activity.textContent = form.dataset.busy || "Procesando solicitud…";
    }
  });
}
