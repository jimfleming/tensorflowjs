var ffi = require('ffi');
var ref = require('ref');
var ArrayType = require('ref-array');

var TF_DataTypeEnum = exports.TF_DataTypeEnum = {
  TF_FLOAT: 1,
  TF_DOUBLE: 2,
  TF_INT32: 3,
  TF_UINT8: 4,
  TF_INT16: 5,
  TF_INT8: 6,
  TF_STRING: 7,
  TF_COMPLEX: 8,
  TF_INT64: 9,
  TF_BOOL: 10,
  TF_QINT8: 11,
  TF_QUINT8: 12,
  TF_QINT32: 13,
  TF_BFLOAT16: 14,
};

var TF_DataTypeStr = exports.TF_DataTypeStr = {
  1: 'TF_FLOAT',
  2: 'TF_DOUBLE',
  3: 'TF_INT32',
  4: 'TF_UINT8',
  5: 'TF_INT16',
  6: 'TF_INT8',
  7: 'TF_STRING',
  8: 'TF_COMPLEX',
  9: 'TF_INT64',
  10: 'TF_BOOL',
  11: 'TF_QINT8',
  12: 'TF_QUINT8',
  13: 'TF_QINT32',
  14: 'TF_BFLOAT16',
};

var TF_CodeEnum = exports.TF_CodeEnum = {
  TF_OK: 0,
  TF_CANCELLED: 1,
  TF_UNKNOWN: 2,
  TF_INVALID_ARGUMENT: 3,
  TF_DEADLINE_EXCEEDED: 4,
  TF_NOT_FOUND: 5,
  TF_ALREADY_EXISTS: 6,
  TF_PERMISSION_DENIED: 7,
  TF_RESOURCE_EXHAUSTED: 8,
  TF_FAILED_PRECONDITION: 9,
  TF_ABORTED: 10,
  TF_OUT_OF_RANGE: 11,
  TF_UNIMPLEMENTED: 12,
  TF_INTERNAL: 13,
  TF_UNAVAILABLE: 14,
  TF_DATA_LOSS: 15,
  TF_UNAUTHENTICATED: 16,
};

var TF_CodeStr = exports.TF_CodeStr = {
  0: 'TF_OK',
  1: 'TF_CANCELLED',
  2: 'TF_UNKNOWN',
  3: 'TF_INVALID_ARGUMENT',
  4: 'TF_DEADLINE_EXCEEDED',
  5: 'TF_NOT_FOUND',
  6: 'TF_ALREADY_EXISTS',
  7: 'TF_PERMISSION_DENIED',
  8: 'TF_RESOURCE_EXHAUSTED',
  9: 'TF_FAILED_PRECONDITION',
  10: 'TF_ABORTED',
  11: 'TF_OUT_OF_RANGE',
  12: 'TF_UNIMPLEMENTED',
  13: 'TF_INTERNAL',
  14: 'TF_UNAVAILABLE',
  15: 'TF_DATA_LOSS',
  16: 'TF_UNAUTHENTICATED',
};

var TF_DataType = 'int';
var TF_Code = 'int';

var TF_Status = 'void';
var TF_Tensor = 'void';
var TF_SessionOptions = 'void';
var TF_Session = 'void';

var TF_StatusPtr = ref.refType(TF_Status);
var TF_TensorPtr = ref.refType(TF_Tensor);
var TF_SessionOptionsPtr = ref.refType(TF_SessionOptions);
var TF_SessionPtr = ref.refType(TF_Session);

var TF_TensorArray = exports.TF_TensorArray = ArrayType(TF_TensorPtr);
var StringArray = exports.StringArray = ArrayType('string');
var LongLongArray = exports.LongLongArray = ArrayType('longlong');
var FloatArray = exports.FloatArray = ArrayType('float');

var libtensorflow = ffi.Library('libtensorflow', {
  'TF_NewStatus': [TF_StatusPtr, [ ]],
  'TF_DeleteStatus': ['void', [ TF_StatusPtr ]],
  'TF_SetStatus': ['void', [ TF_StatusPtr, TF_Code, 'string' ]],
  'TF_GetCode': [TF_Code, [ TF_StatusPtr ]],
  'TF_Message': ['string', [ TF_StatusPtr ]],

  'TF_NewTensor': [TF_TensorPtr, [
    TF_DataType,
    LongLongArray, 'int',
    'void*', 'size_t',
    'void*', 'void*'
  ]],
  'TF_DeleteTensor': ['void', [ TF_TensorPtr ]],
  'TF_TensorType': [TF_DataType, [ TF_TensorPtr ]],
  'TF_NumDims': ['int', [ TF_TensorPtr ]],
  'TF_Dim': ['longlong', [ TF_TensorPtr, 'int' ]],
  'TF_TensorByteSize': ['size_t', [ TF_TensorPtr ]],
  'TF_TensorData': ['void*', [ TF_TensorPtr ]],

  'TF_NewSessionOptions': [TF_SessionOptionsPtr, [ ]],
  'TF_SetTarget': ['void', [ TF_SessionOptionsPtr, 'string' ]],
  'TF_SetConfig': ['void', [ TF_SessionOptionsPtr, 'void*', 'size_t', TF_StatusPtr ]],
  'TF_DeleteSessionOptions': ['void', [ TF_SessionOptionsPtr ]],

  'TF_NewSession': [TF_SessionPtr, [ TF_SessionOptionsPtr, TF_StatusPtr ]],
  'TF_CloseSession': ['void', [ TF_SessionPtr, TF_StatusPtr ]],
  'TF_DeleteSession': ['void', [ TF_SessionPtr, TF_StatusPtr ]],
  'TF_ExtendGraph': ['void', [ TF_SessionPtr, 'void*', 'size_t', TF_StatusPtr ]],

  'TF_Run': ['void', [
    TF_SessionPtr,
    StringArray, TF_TensorArray, 'int',
    StringArray, TF_TensorArray, 'int',
    StringArray, 'int',
    TF_StatusPtr
  ]],
});

// Default destructor for Buffer-backed tensors. It's required by TF but a no-op since Buffers are GC'd.
exports.TF_Destructor = ffi.Callback('void', ['void*', 'size_t', 'void*'], function(data, len, arg) {});

// Throw an error if the status isn't TF_OK
exports.TF_CheckOK = function(status) {
  var code = libtensorflow.TF_GetCode(status);
  if (code !== TF_CodeEnum.TF_OK) {
    throw new Error(TF_Message(status));
  }
}

// TF_TensorData just returns a point to the data but the buffer isn't initialized to the long of the data for us. So we need to expand the buffer with reinterpret then set the correct type so that deref knows how to cast the buffer.
exports.TF_ReadTensorData = function(tensor, size, type) {
  var ptr = libtensorflow.TF_TensorData(tensor);
  ptr = ptr.reinterpret(size, 0);
  ptr.type = type;
  return ptr.deref();
}

for (var name in libtensorflow) {
  if (libtensorflow.hasOwnProperty(name)) {
    exports[name] = libtensorflow[name];
  }
}
