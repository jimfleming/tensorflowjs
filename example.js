var fs = require('fs');
var path = require('path');
var tf = require('./index.js');

// Initialize session
var opts = tf.TF_NewSessionOptions();
var status = tf.TF_NewStatus();
var sess = tf.TF_NewSession(opts, status)
tf.TF_CheckOK(status);

// Load graph
var graph_def = fs.readFileSync('graph.pb');
tf.TF_ExtendGraph(sess, graph_def, graph_def.length, status);
tf.TF_CheckOK(status);

// Initialize tensors
var aDims = new tf.LongLongArray(0);
var aData = new tf.FloatArray([3.0]);
var aTensor = tf.TF_NewTensor(tf.TF_DataTypeEnum.TF_FLOAT,
  aDims, aDims.length,
  aData.buffer, aData.buffer.length,
  tf.TF_Destructor, null);

var bDims = new tf.LongLongArray(0);
var bData = new tf.FloatArray([2.0]);
var bTensor = tf.TF_NewTensor(tf.TF_DataTypeEnum.TF_FLOAT,
  bDims, bDims.length,
  bData.buffer, bData.buffer.length,
  tf.TF_Destructor, null);

// Run graph
var input_names = new tf.StringArray(["a", "b"]);
var output_names = new tf.StringArray(["c"]);
var target_names = new tf.StringArray(0);

var inputs = new tf.TF_TensorArray([aTensor, bTensor]);
var outputs = new tf.TF_TensorArray(1);

tf.TF_Run(sess,
  input_names, inputs, input_names.length,
  output_names, outputs, output_names.length,
  target_names, target_names.length,
  status);
tf.TF_CheckOK(status);

// Read result
var c = outputs[0];

var type = tf.TF_TensorType(c);
var dims = tf.TF_NumDims(c);
var size = tf.TF_TensorByteSize(c);
var val = tf.TF_ReadTensorData(c, size, 'float');

console.log('c type', tf.TF_DataTypeStr[type], type);
console.log('c dims', dims);
console.log('c size', size);
console.log('c val', val);

// Close session
tf.TF_CloseSession(sess, status);
tf.TF_CheckOK(status);
