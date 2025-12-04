document.getElementById("uploadBtn").addEventListener("click", async () => {
    const input = document.getElementById("fileInput");
    const file = input.files[0];

    if (!file) {
        alert("Upload file dulu");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    const resultDiv = document.getElementById("result");
    resultDiv.innerHTML = "<p>Checking...</p>";

    const response = await fetch("/check", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    if (data.error) {
        resultDiv.innerHTML = `<p class="text-danger">${data.error}</p>`;
        return;
    }

    let pct = (data.score * 100).toFixed(2);

    let status = "Tidak Mirip";
    if (data.score > 0.90) status = "Plagiarisme Tinggi";
    else if (data.score > 0.80) status = "Mirip / Parafrase";
    else if (data.score > 0.70) status = "Mirip Topik";

    resultDiv.innerHTML = `
        <h5>Hasil:</h5>
        <p><strong>Similarity:</strong> ${pct}%</p>
        <p><strong>Status:</strong> ${status}</p>
        <p><strong>Nearest Doc ID:</strong> ${data.nearest_doc_id}</p>
    `;
});
