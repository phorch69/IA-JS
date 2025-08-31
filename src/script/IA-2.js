// TensorFlow.js
import * as tf from '@tensorflow/tfjs';

let model; // Declaramos el modelo fuera para usarlo en todo el flujo.

async function crearYEntrenarModelo() {
  // Datos de entrenamiento
  const celsius = tf.tensor1d([-40, -10, 0, 8, 15, 22, 38], 'float32');
  const fahrenheit = tf.tensor1d([-40, 14, 32, 46, 59, 72, 100], 'float32');

  // Crear el modelo
  model = tf.sequential();

  // Capa oculta
  model.add(tf.layers.dense({
    units: 1, // Solo 1 unidad para hacer el modelo más eficiente
    inputShape: [1]
  }));

  // Capa de salida
  model.add(tf.layers.dense({
    units: 1
  }));

  // Compilar el modelo
  model.compile({
    optimizer: tf.train.adam(0.1),
    loss: 'meanSquaredError'
  });

  // Entrenar el modelo
  await model.fit(celsius, fahrenheit, {
    epochs: 100, // Reducimos las épocas a 100
    verbose: 0  // No mostrar detalles de cada época
  });
}

// Hacer la predicción con el modelo entrenado
async function predecirFahrenheit(celsiusValue) {
  // Asegurarnos de que el modelo esté entrenado antes de predecir
  if (!model) {
    await crearYEntrenarModelo();
  }

  const input = tf.tensor2d([celsiusValue], [1, 1]);
  const prediction = model.predict(input);

  return prediction.dataSync()[0];
}

// Función para actualizar el valor de grados Celsius
function cambiarCelsius() {
  const celsiusValue = parseFloat(document.getElementById('celsius').value);
  document.getElementById('lbl-celsius').innerText = celsiusValue;

  // Obtener la predicción y actualizar el resultado
  predecirFahrenheit(celsiusValue).then((fahrenheitValue) => {
    document.getElementById('resultado').innerText = `${celsiusValue} grados Celsius son ${fahrenheitValue.toFixed(2)} grados Fahrenheit`;
  });
}

// Inicializar la página solo una vez
crearYEntrenarModelo().then(() => {
  cambiarCelsius(); // Para que cargue el valor inicial al cargar la página
});