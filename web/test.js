const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");


const stream = canvas.captureStream();
const chunks = [];

var rmsh = ['r', 'm', 's', 'h'];
var Baw = ['B', 'a', 'w'];

var rowCount = 1;

function addRow() {
    rowCount++;
    var table = document.getElementById("myTable");
    var row = table.insertRow();
    row.id = rowCount;
    var rmshParamsCell = row.insertCell(0);
    var BParamsCell = row.insertCell(1);
    var aParamsCell = row.insertCell(2);
    var wParamsCell = row.insertCell(3);
    // var addBInputCell = row.insertCell(4);
    var deleteRowCell = row.insertCell(4);

    for (var i = 0; i < rmsh.length; i++) {
        myAddInputInThisRow(row, 0, rmsh[i], false);
        // let inputContainer = document.createElement("div");
        // inputContainer.className = "inputContainer";
        // rmshParamsCell.appendChild(inputContainer);
        // let newLbl = document.createElement("label");
        // newLbl.textContent = rmsh[i];
        // let newInput = document.createElement("input");
        // newInput.type = "number";
        // // newInput.name = "input" + rowCount + "_" + i;
        // newInput.name = rmsh[i];
        // newInput.value = 0.2;
        // inputContainer.appendChild(newLbl);
        // inputContainer.appendChild(newInput);
    }

    // let inputContainer = document.createElement("div");
    // inputContainer.className = "inputContainer";
    // rmshParamsCell.appendChild(inputContainer);
    // let newLbl = document.createElement("label");
    // newLbl.textContent = 'B';
    // let newInput = document.createElement("input");
    // newInput.type = "number";
    // // newInput.name = "input" + rowCount + "_" + i;
    // newInput.name = 'B';
    // newInput.value = 0.2;
    // inputContainer.appendChild(newLbl);
    // inputContainer.appendChild(newInput);

    for (var k = 0; k < Baw.length; k++) {

        let addInputButton = document.createElement("button");
        addInputButton.type = "button";
        addInputButton.textContent = "Add " + Baw[k];
        addInputButton.onclick = function() { 
            addInput(this, k + 1, Baw[k]); 
        }
        
        row.cells[k + 1].appendChild(addInputButton);
        
    }

    
    let deleteRowButton = document.createElement("button");
    deleteRowButton.type = "button";
    deleteRowButton.textContent = "Delete Row";
    deleteRowButton.onclick = function() { deleteRow(this); };
    deleteRowCell.appendChild(deleteRowButton);

    return row;
}

function deleteRow(btn) {
  var row = btn.parentNode.parentNode;
  row.parentNode.removeChild(row);
}

function myAddInputInThisRow(row, cell_n, p, removable = true) {

    let cell = row.cells[cell_n];

    let inputContainer = document.createElement("div");
    inputContainer.className = "inputContainer";
    cell.appendChild(inputContainer);
    let newLbl = document.createElement("label");
    newLbl.textContent = p;
    let newInput = document.createElement("input");
    newInput.type = "number";
    // newInput.name = "input" + rowCount + "_" + i;
    newInput.name = p;
    newInput.value = 0.2;
    inputContainer.appendChild(newLbl);
    inputContainer.appendChild(newInput);

    if (removable) {
        let removeButton = document.createElement("button");
        removeButton.type = "button";
        removeButton.innerHTML = "Remove Input";
        removeButton.onclick = function() { removeInput(this); };
        inputContainer.appendChild(removeButton);
    }

    return newInput;
}
  

function addInput(btn, cell_n, p) {
  var row = btn.parentNode.parentNode;
  myAddInputInThisRow(row, cell_n, p);
}

function addInputInThisRow(row) {
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
  return newInput;
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

    var rInputs = [];
    var mInputs = [];
    var sInputs = [];
    var hInputs = [];
    var BInputs = [];
    var aInputs = [];
    var wInputs = [];

    var rowData = [];

    for (var i = 0; i < rows.length; i++) {
      var inputs = rows[i].getElementsByTagName("input");
      if (inputs[0]) rInputs.push(inputs[0].value);
      if (inputs[1]) mInputs.push(inputs[1].value);
      if (inputs[2]) sInputs.push(inputs[2].value);
      if (inputs[3]) hInputs.push(inputs[3].value);

      var rowValues = [];
      for (var j = 4; j < inputs.length; j++) {
        rowValues.push(inputs[j].value);
      }
      rowData.push(rowValues);

    }

    console.log({ firstInputs: rInputs, secondInputs: mInputs, thirdInputs: sInputs, fourthInputs: hInputs });
    console.log(rowData);

    params = {'r': rInputs, 'm': mInputs, 's': sInputs, 'h': hInputs, "B": rowData};
    return params;
}



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


async function step() {
    stepNTxt.textContent = parseInt(stepNTxt.textContent) + 1;
    await eel.step()();
    await updateWorldDisplay();
}

async function play() {
    while (is_playing) {
        await step();
    }
}

const playBtn = document.querySelector(".play-btn"),
nextStepBtn = document.querySelector(".next-step-btn"),
restartBtn = document.querySelector(".restart-btn"),
saveVideoBtn = document.querySelector(".save-video-btn"),
saveStateBtn = document.querySelector(".save-state-btn"),
loadStateBtn = document.querySelector(".load-state-btn"),
versionMenu = document.querySelector(".version-selector"),
versionLoadingTxt = document.querySelector(".version-loading-txt"),
stepNTxt = document.querySelector("#step-n-txt");

stepNTxt.textContent = 0;

var versionSelected;

eel.expose(getKernelParamsFromWeb);
function getKernelParamsFromWeb() {
    
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



async function getParamsFromPython() {

    versionLoadingTxt.innerText = "Compiling...";

    table = document.getElementById("myTable");
    table.innerHTML = "";

    var data = await eel.getParameters()();
    
    document.querySelector(".version-selector").value = data["version"];
    document.querySelector(".size").value = data["size"];
    document.querySelector(".num-channels").value = data["numChannels"];
    document.querySelector(".seed").value = data["seed"];
    document.querySelector(".theta").value = data["theta"];
    document.querySelector(".dd").value = data["dd"];
    document.querySelector(".dt").value = data["dt"];
    document.querySelector(".sigma").value = data["sigma"];

    for (var i = 0; i < data["r"].length; i++) {
        
        let row = addRow();
        rmsh.forEach(k =>{
            
            // let inputs = row.cells[0].getElementsByTagName("input");
            let input = row.cells[0].querySelector("input[name='" + k + "']");
            input.value = data[k][i];
        })

        for (var k = 0; k < Baw.length; k++) {

            for (var j = 0; j < data[Baw[k]][i].length; j++) {
                
                // let input = row.cells[0].querySelector("input[name='" + j + "']");
                input = myAddInputInThisRow(row, k + 1, Baw[k]);
                input.value = data[Baw[k]][i][j];
                
            }
        }
    }
    
    updateWorldDisplay();
    versionLoadingTxt.innerText = "";
}

async function setParamsInPython() {

    versionLoadingTxt.innerText = "Compiling...";

    let data = {};
    
    data["version"] = document.querySelector(".version-selector").value;
    data["size"] = parseInt(document.querySelector(".size").value);
    data["numChannels"] = parseInt(document.querySelector(".num-channels").value);
    data["seed"] = parseInt(document.querySelector(".seed").value);
    data["theta"] = parseFloat(document.querySelector(".theta").value);
    data["dd"] = parseInt(document.querySelector(".dd").value);
    data["dt"] = parseFloat(document.querySelector(".dt").value);
    data["sigma"] = parseFloat(document.querySelector(".sigma").value);

    // let kernelParmas = submitForm();

    for (var k = 0; k < rmsh.length; k++) {
        data[rmsh[k]] = [];
    }
    
    for (var k = 0; k < Baw.length; k++) {
        data[Baw[k]] = [];
    }

    var rows = document.getElementsByTagName("tr");

    for (var i = 0; i < rows.length; i++) {
        
        let row = rows[i];


        rmsh.forEach(k =>{
            
            // let inputs = row.cells[0].getElementsByTagName("input");
            let input = row.cells[0].querySelector("input[name='" + k + "']");
            data[k].push(parseFloat(input.value));
        })

        for (var k = 0; k < Baw.length; k++) {

            // data[Baw[k]].push([]);
            // console.log(data);
            inputs = row.cells[k + 1].getElementsByTagName("input");

            let B = [];

            for (var j = 0; j < inputs.length; j++) {
                
                // let input = row.cells[0].querySelector("input[name='" + j + "']");
                B.push(parseFloat(inputs[j].value));
                
            }
            data[Baw[k]].push(B);

        }
    }
    
    await eel.setParameters(data)();
    updateWorldDisplay();
    getParamsFromPython();
    versionLoadingTxt.innerText = "";
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

// versionMenu.addEventListener("change", compile)

playBtn.addEventListener("click", () => {

    is_playing = !is_playing;

    if (is_playing) {
        play();
        playBtn.innerText = "STOP";
    } else {
        playBtn.innerText = "PLAY";
    }
    
})

nextStepBtn.addEventListener("click", async () => {

    is_playing = false;
    step();
    playBtn.innerText = "PLAY";
    
})


restartBtn.addEventListener("click", () => {
    is_playing = false;
    playBtn.innerText = "PLAY";
    setParamsInPython();
    stepNTxt.textContent = 0;
    // getKernelParamsFromWeb();
})

saveVideoBtn.addEventListener("click", () => {

    let nSteps = parseInt(document.querySelector(".sec-to-save").value);
    // eel.saveNStepsToVideo(nSteps);
    mediaRecorder.start();
    setTimeout(function() {
        mediaRecorder.stop();
    }, nSteps * 1000);
})


saveStateBtn.addEventListener("click", () => {
    eel.saveParameterState()();
})

loadStateBtn.addEventListener("click", async () => {
    await eel.loadParameterState()();
    getParamsFromPython();
    updateWorldDisplay();
})
















// for (var i = 0; i < 9; i++) {
//     addRow();
// }
// $(document).ready(async function() {
//     await getParamsFromPython();
//     await setParamsInPython();
// });
getParamsFromPython();
// setParamsInPython();





