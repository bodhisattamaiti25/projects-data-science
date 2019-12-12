import React from 'react';
import * as tf from '@tensorflow/tfjs';
import ClassIndicesMap from './ClassIndicesMap';

const ModelPath = '/model/model.json';
class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isModelLoaded:false,
      model:null,
      imgSrc:'',
      classificationResults: []
    };
  }

  componentDidMount(){
    console.log('the component has successfully mounted');
    this.loadModel();
  }

  loadModel = async () => {
    try {
      //first check if the model exists in IndexedDB1
      const model = await tf.loadLayersModel('indexeddb://plant_health_diagnosis_model');
      console.log('model successfully loaded from indexed db')
      this.setState({model, isModelLoaded: true});
    } catch(error) {
      console.log('error', error)
      console.log('the model does not exist in IndexedDB yet.')
      const model = await tf.loadLayersModel(ModelPath);
      console.log('model successfully loaded from model path')
      this.setState({model, isModelLoaded: true});
      await model.save('indexeddb://plant_health_diagnosis_model');
      console.log('model successfully saved in indexed db')
    }
  }

  handleImgChange = (evt) => {
    console.log('on img change', evt.target.files[0])
    const imgSrc = URL.createObjectURL(evt.target.files[0]);
    this.setState({imgSrc});
  }

  handleImgLoad = () => {
    console.log('Image has successfully loaded')
    this.setState({classificationResults:[]});
  }

  handleImgClassification = async () => {
    console.log('predict button clicked')
    const img = document.getElementById('plantHealthImg');
    console.log('img', img)
    const offset = tf.scalar(127.5);
    const imgTensor = tf.browser.fromPixels(img).resizeNearestNeighbor([224,224]).toFloat();
    const normalizedImgTensor = imgTensor.sub(offset).div(offset).expandDims();
    const {model} = this.state;
    const probabilities = await model.predict(normalizedImgTensor).data();
    console.log('probabilities', probabilities)
    this.displayResults(probabilities)
  }

  displayResults = (probabilities) => {
    const results = [];
    for(let i=0; i<probabilities.length; i++){
      results.push({label: ClassIndicesMap.get(i), val: probabilities[i]})
    }
    results.sort((prev, next) => {
      return next['val'] - prev['val'];
    })
    console.log('prediction results', results)
    this.setState({classificationResults: results.slice(0,5)});
  }

  render() {
    return (
      <div>
        {
          this.state.isModelLoaded 
          ? <>
              <div style={{border:'1px dotted blue', height:'264px', width:'264px'}}>
                <img 
                  id='plantHealthImg'
                  src={this.state.imgSrc} 
                  alt='Image Box' 
                  width='224' 
                  height='224' 
                  onLoad={this.handleImgLoad}
                  style={{padding:'20px'}}
                />
              </div>
              <h4>Upload an Image</h4>
              <input type='file' name='plantImgFile' accept='image/*' onChange={this.handleImgChange}/>
              <button onClick={this.handleImgClassification}>Predict</button>
              <div>
                <h2>Predictions</h2>
                {this.state.classificationResults.map((plant, i)=>{
                  return <div key={i}>
                          <span>{plant.label}: </span>
                          <span>{plant.val}</span>
                         </div>
                })}
              </div>
            </>
          : <h2>Model is Loading</h2>
        }
      </div>
    )
  }
}

export default App;
