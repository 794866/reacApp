import React from 'react';
import * as tf from '@tensorflow/tfjs';
import './app.css';
import exampleImage from './example.png';
import {loadGraphModel} from '@tensorflow/tfjs-converter';


const MODEL_URL = 'https://github.com/794866/TFM/blob/main/cnn/modelos/binario/model.json';

async function runModel() {
    //const model = await tf.loadGraphModel('https://raw.githubusercontent.com/daved01/tensorflowjs-web-app-demo/main/models/fullyConvolutionalModelTfjs/model.json');
    
    //MY DIR MODELS
    //const model = await loadGraphModel(MODEL_URL);
    const model = await tf.loadLayersModel('https://github.com/794866/TFM/blob/main/cnn/modelos/binario/model.json');



    console.log(model);
    if(model != null){
        console.log("modelo cargado...");
    }

    // Get content image
    let image = new Image(256,256);
    image.src = exampleImage;

    // Convert image to tensor and add batch dimension
    let tfTensor = tf.browser.fromPixels(image);    
    tfTensor = tfTensor.div(255.0);
    tfTensor = tfTensor.expandDims(0);
    tfTensor = tfTensor.cast("float32");
    
    // Run image through model
    const pred = model.predict(tfTensor);

    console.log(pred);
    
}


function App(props) {  
    return (
        <div className="main">
            <h1>App</h1>
            <div className="imageContainer">
                <img className="myImage" src={exampleImage} alt="Image" height={256} width={256} />
                <canvas className="myImage" id="canvas" width={256} height={256}> </canvas>
            </div>  
            <div className="myButtonPos">
                <button className="myButton" onClick={runModel}>Run model</button>
            </div>                              
        </div>
    );
}

export default App;