async function fetchDashboardData() {
  const res = await fetch("/dashboard-data", { method: "GET" });
  if (!res.ok) throw new Error("Failed to fetch /dashboard-data");
  return await res.json();
}

function riskToPercent(prob01) {
  return (Number(prob01) * 100).toFixed(2) + "%";
}

document.addEventListener("DOMContentLoaded", async () => {
  try {
    const data = await fetchDashboardData();

    document.getElementById("totalPred").textContent = data.total ?? 0;
    document.getElementById("posCount").textContent = data.positive_count ?? 0;
    document.getElementById("posRate").textContent =
      (Number(data.positive_rate ?? 0)).toFixed(2) + "%";

    const labels = ["Low", "Medium", "High"];
    const values = labels.map((l) => data.risk_counts?.[l] ?? 0);

    const ctx = document.getElementById("riskChart");
    new Chart(ctx, {
      type: "doughnut",
      data: {
        labels,
        datasets: [
          {
            data: values,
            backgroundColor: ["#198754", "#ffc107", "#dc3545"],
          },
        ],
      },
      options: {
        responsive: true,
      },
    });

    const recentBody = document.getElementById("recentBody");
    if (data.recent && data.recent.length) {
      recentBody.innerHTML = data.recent
        .map((r) => {
          const created = r.created_at ? r.created_at.toString() : "";
          return `
            <tr>
              <td>${r.id}</td>
              <td>${r.age}</td>
              <td>${r.glucose}</td>
              <td>${r.blood_pressure}</td>
              <td>${r.bmi}</td>
              <td>${r.result}</td>
              <td>${riskToPercent(r.probability)}</td>
              <td>${created}</td>
            </tr>
          `;
        })
        .join("");
    } else {
      recentBody.innerHTML =
        '<tr><td colspan="8" class="text-muted text-center">No predictions yet.</td></tr>';
    }
  } catch (e) {
    console.error(e);
    alert("Dashboard load failed. Please login again or check backend.");
  }
});

