// ================= LOGIN =================
async function login(){

let username = document.getElementById("username").value;
let password = document.getElementById("password").value;
let selectedAnalysis = null;

let response = await fetch(
`http://localhost:8000/auth/login?username=${username}&password=${password}`,
{ method:"POST" }
);

let data = await response.json();

localStorage.setItem("username", username);

if(data.role === "doctor"){
window.location.href="doctor_dashboard.html";
}else{
localStorage.setItem("patient", username);
window.location.href="patient_dashboard.html";
}

}

// ================= REGISTER =================
async function register(event){

event.preventDefault();

let username = document.getElementById("username").value;
let password = document.getElementById("password").value;
let role = document.getElementById("role").value;

let response = await fetch(
`http://localhost:8000/auth/register?username=${username}&password=${password}&role=${role}`,
{ method:"POST" }
);

let data = await response.json();

alert(data.message);
window.location.href="login.html";

}

// ================= DASHBOARD =================
async function loadDoctorDashboard(){

let doctor = localStorage.getItem("username");

let response = await fetch(
`http://localhost:8000/api/doctor/dashboard?doctor_username=${doctor}`
);

let data = await response.json();

document.getElementById("stats").innerHTML = `
Total análisis: ${data.total_analyses}<br>
Pacientes únicos: ${data.unique_patients}<br>
Diagnósticos confirmados: ${data.confirmed_diagnoses}
`;

}

// ================= PACIENTES =================
async function loadPatients(){

let response = await fetch(
`http://localhost:8000/api/doctor/all-patients`
);

let patients = await response.json();

console.log("PACIENTES:", patients);

let list = document.getElementById("patients");

list.innerHTML = "";

if(patients.length === 0){
list.innerHTML = "<p>No hay pacientes registrados</p>";
return;
}

patients.forEach(p => {

let li = document.createElement("li");

li.innerHTML = `
<b>${p.username}</b><br>
<button onclick="selectPatient('${p.username}')">Analizar</button>
<button onclick="viewPatient('${p.username}')">Historial</button>
<hr>
`;

list.appendChild(li);

});

}

// ================= SELECCIONAR PACIENTE =================
function selectPatient(patient){

localStorage.setItem("patient", patient);
window.location.href = "analysis.html";

}

// ================= HISTORIAL =================
function viewPatient(patient){

localStorage.setItem("patient", patient);
window.location.href = "history.html";

}

async function loadHistory(){

let doctor = localStorage.getItem("username");
let patient = localStorage.getItem("patient");

let response = await fetch(
`http://localhost:8000/api/doctor/patient-history?doctor_username=${doctor}&patient_username=${patient}`
);

let data = await response.json();

let table = document.getElementById("historyTable");

table.innerHTML = "";

data.forEach(a => {

let row = `
<tr>
<td>${a.created_at}</td>
<td>${a.diagnosis}</td>
<td>
<button onclick="viewAnalysis(${a.id}, '${a.report}')">Ver</button>
<button onclick="downloadPDF(${a.id})">PDF</button>
</td>
</tr>
`;

table.innerHTML += row;

});

}

// ================= PDF =================
function downloadPDF(id){

let username = localStorage.getItem("username");

window.open(
`http://localhost:8000/api/analysis/${id}/export-pdf?username=${username}`
);

}

// ================= ANALYSIS =================
function showPatient(){

let patient = localStorage.getItem("patient");

document.getElementById("patientName").innerHTML =
"Paciente: " + patient;

}

function previewImage(){

let file = document.getElementById("imageInput").files[0];

let reader = new FileReader();

reader.onload = function(e){
document.getElementById("preview").src = e.target.result;
}

reader.readAsDataURL(file);

}

async function uploadImage(){

let file = document.getElementById("imageInput").files[0];

let doctor = localStorage.getItem("username");
let patient = localStorage.getItem("patient");

if(!file){
alert("Selecciona una imagen");
return;
}

let formData = new FormData();
formData.append("file", file);

let response = await fetch(
`http://localhost:8000/api/analyze?username=${doctor}&patient_username=${patient}`,
{
method:"POST",
body:formData
});

let data = await response.json();

document.getElementById("result").innerHTML =
"<b>Diagnóstico IA:</b> " + data.diagnostico_ia + "<br><br>" +
"<b>Reporte:</b> " + data.reporte_generado;

}
function viewAnalysis(id, report){

selectedAnalysis = id;

let username = localStorage.getItem("username");

// Imagen
document.getElementById("analysisImage").src =
`http://localhost:8000/api/analysis-image/${id}?username=${username}`;

// Reporte
document.getElementById("analysisReport").innerHTML =
"<b>Reporte:</b><br>" + report;

}
async function confirmDiagnosis(){

let doctor = localStorage.getItem("username");
let diagnosis = document.getElementById("confirmInput").value;

if(!selectedAnalysis){
alert("Selecciona un estudio primero");
return;
}

await fetch(
`http://localhost:8000/api/analysis/${selectedAnalysis}/confirm-diagnosis?doctor_username=${doctor}&confirmed_diagnosis=${diagnosis}`,
{
method:"PUT"
});

alert("Diagnóstico confirmado");

}
async function saveNotes(){

let doctor = localStorage.getItem("username");
let notes = document.getElementById("notes").value;

if(!selectedAnalysis){
alert("Selecciona un estudio primero");
return;
}

await fetch(
`http://localhost:8000/api/analysis/${selectedAnalysis}/doctor-notes?doctor_username=${doctor}&notes=${notes}`,
{
method:"PUT"
});

alert("Notas guardadas");

}
// ================= PANEL PACIENTE =================
async function loadPatientDashboard(){

let patient = localStorage.getItem("username");

document.getElementById("patientName").innerHTML =
"Paciente: " + patient;

let response = await fetch(
`http://localhost:8000/api/patient-history?username=${patient}`
);

let data = await response.json();

let table = document.getElementById("patientHistory");

table.innerHTML = "";

data.forEach(a => {

let row = `
<tr>
<td>${a.created_at}</td>
<td>${a.diagnosis}</td>
<td>
<button onclick="viewPatientAnalysis(${a.id}, '${a.report}')">Ver</button>
<button onclick="downloadPDF(${a.id})">PDF</button>
</td>
</tr>
`;

table.innerHTML += row;

});

}
function viewPatientAnalysis(id, report){

let username = localStorage.getItem("username");

// Imagen
document.getElementById("analysisImage").src =
`http://localhost:8000/api/analysis-image/${id}?username=${username}`;

// Reporte
document.getElementById("analysisReport").innerHTML =
"<b>Reporte:</b><br>" + report;

}
function logout(){
localStorage.clear();
window.location.href = "index.html";
}