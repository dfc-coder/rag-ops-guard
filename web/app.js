const activity = document.querySelector("#activity");

const messages = {
  "/query": "Procesando consulta…",
  "/documents": "Ingeriendo documento…",
  "/demo/load": "Recargando corpus demo…",
};

for (const form of document.querySelectorAll("form")) {
  form.addEventListener("submit", () => {
    const path = new URL(form.action, window.location.href).pathname;
    const button = form.querySelector("button[type='submit']");

    if (button) {
      button.disabled = true;
      button.textContent = "Procesando…";
    }

    if (activity) {
      activity.hidden = false;
      activity.textContent = messages[path] || "Procesando solicitud…";
    }
  });
}
