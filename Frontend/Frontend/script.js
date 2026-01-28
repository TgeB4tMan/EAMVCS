async function generate() {
  const text = document.getElementById("text").value;
  const language = document.getElementById("language").value;
  const audio = document.getElementById("audio").files[0];

  if (text.trim() === "") {
    alert("Please enter some text");
    return;
  }

  const formData = new FormData();
  formData.append("text", text);
  formData.append("language", language);
  if (audio) formData.append("audio", audio);

  try {
    const response = await fetch("http://localhost:5000/generate", {
      method: "POST",
      body: formData
    });

    if (!response.ok) throw new Error();

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);

    document.getElementById("player").src = url;
    document.getElementById("download").href = url;

  } catch {
    alert("Backend not connected");
  }
}
