const fileElem = document.getElementById("fileElem");
const statusMsg = document.getElementById("statusMsg");
// Handle file selection
fileElem.addEventListener("change", () => {
  const file = fileElem.files[0];
  if (file) {
    statusMsg.textContent = `✅ Selected: ${file.name}`;
    statusMsg.style.color = "#b16eff";
  } else {
    statusMsg.textContent = "";
  }
});
// Star animation (keep this part as is)
const starContainer = document.getElementById("stars");
for (let i = 0; i < 150; i++) {
  const star = document.createElement("div");
  star.classList.add("star");
  star.style.top = `${Math.random() * 100}%`;
  star.style.left = `${Math.random() * 100}%`;
  star.style.animationDuration = `${1.5 + Math.random() * 3}s`;
  star.style.opacity = Math.random();
  starContainer.appendChild(star);
}