const videoElement = document.getElementsByClassName("input_video")[0];
const canvasElement = document.getElementsByClassName("output_canvas")[0];
const canvasCtx = canvasElement.getContext("2d");
let collecting = false;
let predictionData = [];
let key;

function setup() {
  noCanvas();
  const camera = new Camera(videoElement, {
    onFrame: async () => {
      await hands.send({ image: videoElement });
    },
    width: 1280,
    height: 720,
  });
  camera.start();

  const hands = new Hands({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.1/${file}`;
    },
  });
  hands.setOptions({
    selfieMode: true,
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  
  hands.onResults(onResults);
  videoElement.style.display = "none";
}

function draw() {
  background(220);
}

document.addEventListener('keydown', keyPressed);

function keyPressed(event) {
  if (event.key !== 's') { 
    if (!collecting) {
      key = event.key;
      collecting = true;
      console.log(`${event.key} pressed. Start collecting...`);
      
      setTimeout(() => {
        console.log('Stopping collection...');
        collecting = false;
      }, 5000); 
    }
  } else if (event.key === 's') {
    const labeledData = {
      data: predictionData
    };
    const blob = new Blob([JSON.stringify(labeledData)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'labeledData.json';  
    document.body.appendChild(a);  
    a.click();                     
    
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    console.log('Data saved and download initiated.');

  }
}

function onResults(results) {
  if (collecting) {
    setInterval(() => {
      const handData = results.multiHandLandmarks.map((hand) => 
        Object.values(hand).flat()
      );
      predictionData.push(
        {
          xs: handData.flat(), 
          ys: { "0": key }
        }
      );
    }, 100);

    clearInterval();
  }
  
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(
    results.image,
    0,
    0,
    canvasElement.width,
    canvasElement.height
  );
  if (results.multiHandLandmarks) {
    for (const landmarks of results.multiHandLandmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
        color: "#00FF00",
        lineWidth: 5,
      });
      drawLandmarks(canvasCtx, landmarks, { color: "#FF0000", lineWidth: 2 });
    }
  }
  canvasCtx.restore();
}


