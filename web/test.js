const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");

const stream = canvas.captureStream();
const chunks = [];

const mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });
mediaRecorder.ondataavailable = function(e) {
  chunks.push(e.data);
};
mediaRecorder.onstop = function() {
  const blob = new Blob(chunks, { type: "video/webm" });
  const url = URL.createObjectURL(blob);
  downloadFile(url, "myVideo.webm");
};

function downloadFile(url, filename) {
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
}


async function updateWorldDisplay() {
    
    let A = new Uint8ClampedArray(await eel.getWorld()());
    // console.log(A);
    let sz = Math.sqrt(A.length / 4)
    let img_A = new ImageData(A, sz, sz);

    // let svg = d3.select("span").append("svg")
    //   .attr("width", sz)
    //   .attr("height", sz);
    ctx.createImageData(400, 400);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.putImageData(img_A, 0, 0);

    // svg.append("g")
    // .attr("transform", "translate(0," + sz + ") scale(1,-1)")
    // .append("image")
    // .attr("xlink:href", "data:image/png;base64," + btoa(String.fromCharCode.apply(null, img_A.data)))
    // .attr("width", sz)
    // .attr("height", sz);

}

var is_playing = false;

async function play() {

    while (is_playing) {
        
        await eel.step()()
        await updateWorldDisplay();
    }

    
}

const playBtn = document.querySelector(".play-btn"),
restartBtn = document.querySelector(".restart-btn"),
saveBtn = document.querySelector(".save-btn");
versionMenu = document.querySelector(".version-selector"),
versionLoadingTxt = document.querySelector(".version-loading-txt");

var versionSelected;

eel.expose(getParameters);
function getParameters() {
    
    let size = parseInt(document.querySelector(".size").value),
    numChannels = parseInt(document.querySelector(".num-channels").value),
    seed = parseInt(document.querySelector(".seed").value),
    theta = parseFloat(document.querySelector(".theta").value),
    dd = parseFloat(document.querySelector(".dd").value),
    dt = parseFloat(document.querySelector(".dt").value),
    sigma = parseFloat(document.querySelector(".sigma").value);
    kermel_params = submitForm();
    
    console.log(size);
    return {
        size: size,
        numChannels: numChannels,
        seed: seed,
        theta: theta,
        dd: dd,
        dt: dt,
        sigma: sigma,
        kernel_params: kermel_params
    }
}


function compile() {
    versionSelected = versionMenu.value;
    compile_version(versionSelected);
}

async function compile_version(version) {
    versionLoadingTxt.innerText = "Compiling...";
    await eel.compile_version(version)();
    updateWorldDisplay();
    versionLoadingTxt.innerText = "";
}

versionMenu.addEventListener("change", compile)

playBtn.addEventListener("click", () => {

    is_playing = !is_playing;

    if (is_playing) {
        play();
        playBtn.innerText = "STOP";
    } else {
        playBtn.innerText = "PLAY";
    }
    
})


restartBtn.addEventListener("click", () => {
    is_playing = false;
    playBtn.innerText = "PLAY";
    compile()
})

saveBtn.addEventListener("click", () => {

    let nSteps = parseInt(document.querySelector(".n-steps").value);
    // eel.saveNStepsToVideo(nSteps);
    mediaRecorder.start();
    setTimeout(function() {
        mediaRecorder.stop();
    }, nSteps * 1000);
})










var rowCount = 1;

function addRow() {
  rowCount++;
  var table = document.getElementById("myTable");
  var row = table.insertRow();
  var cell1 = row.insertCell(0);
  var cell2 = row.insertCell(1);
  var cell3 = row.insertCell(2);

  rmsh = ['r', 'm', 's', 'h'];

  for (var i = 0; i < 4; i++) {
    let inputContainer = document.createElement("div");
    inputContainer.className = "inputContainer";
    cell1.appendChild(inputContainer);
    let newLbl = document.createElement("label");
    newLbl.textContent = rmsh[i];
    let newInput = document.createElement("input");
    newInput.type = "number";
    newInput.name = "input" + rowCount + "_" + i;
    newInput.value = 0.2;
    inputContainer.appendChild(newLbl);
    inputContainer.appendChild(newInput);
  }

  var inputContainer = document.createElement("div");
  inputContainer.className = "inputContainer";
  row.cells[0].appendChild(inputContainer);
  var inputCount = row.cells[0].getElementsByClassName("inputContainer").length;
  let newLbl = document.createElement("label");
  newLbl.textContent = 'B';
  var newInput = document.createElement("input");
  newInput.type = "text";
  newInput.name = "input" + rowCount + "_" + inputCount;
  newInput.value = 0.2;
  inputContainer.appendChild(newLbl);
  inputContainer.appendChild(newInput);
  var removeButton = document.createElement("button");
  removeButton.type = "button";
  removeButton.innerHTML = "Remove Input";
  removeButton.onclick = function() { removeInput(this); };
  inputContainer.appendChild(removeButton);

  cell2.innerHTML = "<button type='button' onclick='addInput(this)'>Add Input</button>";
  cell3.innerHTML = "<button type='button' onclick='deleteRow(this)'>Delete Row</button>";
}

function deleteRow(btn) {
  var row = btn.parentNode.parentNode;
  row.parentNode.removeChild(row);
}

function addInput(btn) {
  var row = btn.parentNode.parentNode;
  var inputContainer = document.createElement("div");
  inputContainer.className = "inputContainer";
  row.cells[0].appendChild(inputContainer);
  var inputCount = row.cells[0].getElementsByClassName("inputContainer").length;
  var newInput = document.createElement("input");
  newInput.type = "number";
  newInput.name = "input" + rowCount + "_" + inputCount;
  newInput.value = 0.2;
  inputContainer.appendChild(newInput);
  var removeButton = document.createElement("button");
  removeButton.type = "button";
  removeButton.innerHTML = "Remove Input";
  removeButton.onclick = function() { removeInput(this); };
  inputContainer.appendChild(removeButton);
}

function removeInput(btn) {
  var inputContainer = btn.parentNode;
  inputContainer.parentNode.removeChild(inputContainer);
  var row = btn.parentNode.parentNode.parentNode;
  var inputCount = row.cells[0].getElementsByClassName("inputContainer").length;
  var inputs = row.cells[0].getElementsByTagName("input");
  for (var i = 4; i < inputs.length; i++) {
    inputs[i].name = "input" + rowCount + "_" + (i+1);
  }
}


function submitForm() {
    var table = document.getElementById("myTable");
    var rows = table.getElementsByTagName("tr");

    var firstInputs = [];
    var secondInputs = [];
    var thirdInputs = [];
    var fourthInputs = [];

    var rowData = [];

    for (var i = 0; i < rows.length; i++) {
      var inputs = rows[i].getElementsByTagName("input");
      if (inputs[0]) firstInputs.push(inputs[0].value);
      if (inputs[1]) secondInputs.push(inputs[1].value);
      if (inputs[2]) thirdInputs.push(inputs[2].value);
      if (inputs[3]) fourthInputs.push(inputs[3].value);

      var rowValues = [];
      for (var j = 4; j < inputs.length; j++) {
        rowValues.push(inputs[j].value);
      }
      rowData.push(rowValues);

    }

    console.log({ firstInputs, secondInputs, thirdInputs, fourthInputs });
    console.log(rowData);

    params = {'r': firstInputs, 'm': secondInputs, 's': thirdInputs, 'h': fourthInputs, "B": rowData};
    return params;
}











for (var i = 0; i < 9; i++) {
    addRow();
}
compile();













function addRow2() {
    
    rowCount++;
    var table = document.getElementById("myTable");
    var row = table.insertRow();
    
    for (var i = 0; i < 4; i++) {
        row.insertCell(i);
    }
    var cell_B = row.insertCell(4);

    rmsh = ['r', 'm', 's', 'h'];

    var cellRemoveButton = row.insertCell(5);
    var cellAddButton = row.insertCell(6);
    var inputCount = 7;
    
    for (var p = 0; p < 4; p++) {
        let inputContainer = document.createElement("div");
        inputContainer.className = "inputContainer";
        row.cells[0].appendChild(inputContainer);
        var inputCount = row.cells[p].getElementsByClassName("inputContainer").length;
        var newInput = document.createElement("input");
        newInput.type = "text";
        newInput.name = "input" + rowCount + "_" + inputCount;
        inputContainer.appendChild(newInput);
        var removeButton = document.createElement("button");
        removeButton.type = "button";
        removeButton.innerHTML = "Remove Input";
        removeButton.onclick = function() { removeInput(this); };
        inputContainer.appendChild(removeButton);
    }
}