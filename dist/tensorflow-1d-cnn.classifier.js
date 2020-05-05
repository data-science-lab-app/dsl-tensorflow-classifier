"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var data_science_lab_core_1 = require("data-science-lab-core");
var tf = require("@tensorflow/tfjs");
var tfCore = require("@tensorflow/tfjs-core");
var tfjs_1 = require("@tensorflow/tfjs");
var types_1 = require("@tensorflow/tfjs-core/dist/io/types");
var Tensorflow1dCnnClassifier = /** @class */ (function (_super) {
    __extends(Tensorflow1dCnnClassifier, _super);
    function Tensorflow1dCnnClassifier() {
        var _this = _super.call(this) || this;
        _this.options = new Tensorflow1dCnnClassifierPluginOptions(_this);
        _this.inputs = new Tensorflow1dCnnClassifierPluginInputs(_this);
        _this.data = {};
        return _this;
    }
    Tensorflow1dCnnClassifier.prototype.setInput = function (examples) {
        this.data.inputData = examples;
    };
    Tensorflow1dCnnClassifier.prototype.setOutput = function (examples) {
        this.data.inputLabels = examples;
    };
    Tensorflow1dCnnClassifier.prototype.setRecorderService = function (recorder) {
        this.recorder = recorder;
    };
    Tensorflow1dCnnClassifier.prototype.getInputs = function () {
        return this.inputs;
    };
    Tensorflow1dCnnClassifier.prototype.getOptions = function () {
        return this.options;
    };
    Tensorflow1dCnnClassifier.prototype.finishTraining = function () {
        return false;
    };
    Tensorflow1dCnnClassifier.prototype.saveModel = function (model) {
        return __awaiter(this, void 0, void 0, function () {
            var artifact, json, _a;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        artifact = {
                            modelTopology: model.toJSON({}, false),
                            format: 'layers-model',
                            generatedBy: "TensorFlow.js tfjs-layers v1.7.3",
                            convertedBy: null,
                            userDefinedMetadata: model.getUserDefinedMetadata(),
                            weightSpecs: model.getWeights().map(function (value) {
                                var variable = value;
                                return {
                                    name: variable.name,
                                    shape: variable.shape,
                                    dtype: variable.dtype
                                };
                            })
                        };
                        model.getWeights();
                        _a = {
                            artifact: artifact
                        };
                        return [4 /*yield*/, this.encodeWeights(model.getWeights())];
                    case 1:
                        json = (_a.weights = _b.sent(),
                            _a);
                        return [2 /*return*/, JSON.stringify(json)];
                }
            });
        });
    };
    Tensorflow1dCnnClassifier.prototype.encodeWeights = function (tensors) {
        return __awaiter(this, void 0, void 0, function () {
            var data, totalByteLength, y, offset, out;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, Promise.all(tensors.map(function (t) { return t.data(); }))];
                    case 1:
                        data = _a.sent();
                        totalByteLength = 0;
                        data.forEach(function (datum) {
                            totalByteLength += datum.byteLength;
                        });
                        y = new Uint8Array(totalByteLength);
                        offset = 0;
                        data.forEach(function (x) {
                            y.set(new Uint8Array(x.buffer), offset);
                            offset += x.byteLength;
                        });
                        out = Buffer.from(y.buffer).toString('binary');
                        return [2 /*return*/, out];
                }
            });
        });
    };
    Tensorflow1dCnnClassifier.prototype.decodeWeights = function (weights, specs) {
        return __awaiter(this, void 0, void 0, function () {
            var temp, buffer, out, offset, _i, specs_1, spec, dtype, shape, size, values, dtypeFactor, byteBuffer;
            return __generator(this, function (_a) {
                temp = Buffer.from(weights, 'binary');
                buffer = temp.buffer.slice(temp.byteOffset, temp.byteOffset + temp.byteLength);
                out = [];
                offset = 0;
                for (_i = 0, specs_1 = specs; _i < specs_1.length; _i++) {
                    spec = specs_1[_i];
                    dtype = spec.dtype;
                    shape = spec.shape;
                    size = tfCore.util.sizeFromShape(shape);
                    values = void 0;
                    dtypeFactor = types_1.DTYPE_VALUE_SIZE_MAP[dtype];
                    byteBuffer = buffer.slice(offset, offset + size * dtypeFactor);
                    values = new Float32Array(byteBuffer);
                    offset += size * dtypeFactor;
                    out.push(tfjs_1.tensor(values, shape, dtype));
                }
                return [2 /*return*/, out];
            });
        });
    };
    // async export(minimal: boolean): Promise<string> {
    //     const handler = new TensorflowIOHandler();
    //     // const temp = this.decodeWeights('', []);
    //     // this.data.model.setWeights(temp);
    //     await this.data.model.save(handler);
    //     if (minimal) {
    //         const data: Tensorflow1dCnnClassifierMinimalData = {
    //             labels: this.data.labels,
    //             model_config: handler.json as string
    //         };
    //         return JSON.stringify(data);
    //     } else {
    //         const data: Tensorflow1dCnnClassifierData = {
    //             batchSize: this.data.batchSize,
    //             inputData: this.data.inputData,
    //             inputLabels: this.data.inputLabels,
    //             labels: this.data.labels,
    //             model_config: handler.json as string
    //         };
    //         return JSON.stringify(data);
    //     }
    // }
    Tensorflow1dCnnClassifier.prototype.export = function (minimal) {
        return __awaiter(this, void 0, void 0, function () {
            var weights, specs, data, data;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.encodeWeights(this.data.model.getWeights())];
                    case 1:
                        weights = _a.sent();
                        specs = this.data.model.getWeights().map(function (value) {
                            var variable = value;
                            return {
                                name: variable.name,
                                shape: variable.shape,
                                dtype: variable.dtype
                            };
                        });
                        if (minimal) {
                            data = {
                                labels: this.data.labels,
                                weights: weights,
                                specs: specs,
                                activation: this.data.activation,
                                inputShape: [this.data.inputData[0].length, 1]
                            };
                            return [2 /*return*/, JSON.stringify(data)];
                        }
                        else {
                            data = {
                                labels: this.data.labels,
                                weights: weights,
                                specs: specs,
                                activation: this.data.activation,
                                inputShape: [this.data.inputData[0].length, 1],
                                batchSize: this.data.batchSize,
                                inputData: this.data.inputData,
                                inputLabels: this.data.inputLabels,
                            };
                            return [2 /*return*/, JSON.stringify(data)];
                        }
                        return [2 /*return*/];
                }
            });
        });
    };
    Tensorflow1dCnnClassifier.prototype.import = function (json, minimal) {
        return __awaiter(this, void 0, void 0, function () {
            var data, shape, _a, _b;
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0:
                        data = JSON.parse(json);
                        this.data.labels = data.labels;
                        this.data.activation = data.activation;
                        shape = data.inputShape;
                        this.data.model = tf.sequential();
                        this.data.model.add(tf.layers.conv1d({
                            inputShape: shape,
                            kernelSize: 5,
                            filters: 8,
                            strides: 1,
                            activation: 'relu',
                            kernelInitializer: 'varianceScaling'
                        }));
                        this.data.model.add(tf.layers.maxPooling1d({ poolSize: 2, strides: 2 }));
                        this.data.model.add(tf.layers.conv1d({
                            kernelSize: 5,
                            filters: 16,
                            strides: 1,
                            activation: 'relu',
                            kernelInitializer: 'varianceScaling'
                        }));
                        this.data.model.add(tf.layers.maxPooling1d({ poolSize: 2, strides: 2 }));
                        this.data.model.add(tf.layers.flatten());
                        this.data.model.add(tf.layers.dense({
                            units: this.data.labels.length,
                            kernelInitializer: 'varianceScaling',
                            activation: this.data.activation
                        }));
                        _b = (_a = this.data.model).setWeights;
                        return [4 /*yield*/, this.decodeWeights(data.weights, data.specs)];
                    case 1:
                        _b.apply(_a, [_c.sent()]);
                        if (!minimal) {
                            this.data.labels = data.labels;
                            this.data.batchSize = data.batchSize;
                            this.data.inputData = data.inputData;
                            this.data.inputLabels = data.inputLabels;
                            this.generateTestingData();
                            this.compileModel();
                        }
                        return [2 /*return*/, this];
                }
            });
        });
    };
    // async import(json: string, minimal: boolean): Promise<Tensorflow1dCnnClassifier> {
    //     if (minimal) {
    //         const data = JSON.parse(json) as Tensorflow1dCnnClassifierMinimalData;
    //         this.data.labels = data.labels;
    //         const handler = new TensorflowIOHandler(data.model_config);
    //         this.data.model = await (tf.loadLayersModel(handler, { strict: true })) as tf.Sequential;
    //     } else {
    //         const data = JSON.parse(json) as Tensorflow1dCnnClassifierData;
    //         this.data.labels = data.labels;
    //         this.data.batchSize = data.batchSize;
    //         this.data.inputData = data.inputData;
    //         this.data.inputLabels = data.inputLabels;
    //         const handler = new TensorflowIOHandler(data.model_config);
    //         this.data.model = await (tf.loadLayersModel(handler, { strict: true })) as tf.Sequential;
    //         this.generateTestingData();
    //         this.compileModel();
    //     }
    //     return this;
    // }
    Tensorflow1dCnnClassifier.prototype.compileModel = function () {
        var optimizer = tf.train.adam();
        this.data.model.compile({
            optimizer: optimizer,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy', 'mse'],
        });
    };
    Tensorflow1dCnnClassifier.prototype.test = function (argument) {
        var argumentInput = argument['input'];
        var tfData = tf.tensor2d(argumentInput, [1, argumentInput[0].length]);
        var testInput = tfData.reshape([1, argumentInput[0].length, 1]);
        var testOutput = this.data.model.predict(testInput);
        var output = testOutput.arraySync()[0];
        return {
            'output': [this.data.labels[output.indexOf(Math.max.apply(Math, output))]]
        };
    };
    Tensorflow1dCnnClassifier.prototype.generateTestingData = function () {
        var _a;
        var _this = this;
        var labels = this.data.inputLabels.map(function (value) {
            var label = Array(_this.data.labels.length).fill(0.0);
            label[_this.data.labels.indexOf(value[0])] = 1.0;
            return label;
        });
        var tfData = tf.tensor2d(this.data.inputData, [this.data.inputData.length, this.data.inputData[0].length]);
        var tfLabels = tf.tensor2d(labels, [this.data.inputLabels.length, this.data.labels.length]);
        _a = tf.tidy(function () {
            return [
                tfData.reshape([_this.data.inputData.length, _this.data.inputData[0].length, 1]),
                tfLabels
            ];
        }), this.data.trainX = _a[0], this.data.trainLabels = _a[1];
    };
    Tensorflow1dCnnClassifier.prototype.initialize = function () {
        this.generateTestingData();
        this.data.model = tf.sequential();
        this.data.model.add(tf.layers.conv1d({
            inputShape: [this.data.inputData[0].length, 1],
            kernelSize: 5,
            filters: 8,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));
        this.data.model.add(tf.layers.maxPooling1d({ poolSize: 2, strides: 2 }));
        this.data.model.add(tf.layers.conv1d({
            kernelSize: 5,
            filters: 16,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));
        this.data.model.add(tf.layers.maxPooling1d({ poolSize: 2, strides: 2 }));
        this.data.model.add(tf.layers.flatten());
        this.data.model.add(tf.layers.dense({
            units: this.data.labels.length,
            kernelInitializer: 'varianceScaling',
            activation: this.data.activation
        }));
        this.compileModel();
    };
    Tensorflow1dCnnClassifier.prototype.step = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _this = this;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.data.model.fit(this.data.trainX, this.data.trainLabels, {
                            batchSize: this.data.batchSize,
                            epochs: 1,
                            callbacks: {
                                onBatchEnd: function (batch, logs) {
                                    var _a;
                                    _this.data.model.stopTraining = true;
                                    if (logs) {
                                        (_a = _this.recorder) === null || _a === void 0 ? void 0 : _a.record([
                                            {
                                                label: 'Mean Squared Error',
                                                value: logs['mse'],
                                            },
                                            {
                                                label: 'Accuracy',
                                                value: logs['acc'],
                                            },
                                            {
                                                label: 'Loss',
                                                value: logs['loss'],
                                            }
                                        ]);
                                    }
                                }
                            }
                        })];
                    case 1:
                        (_a.sent());
                        return [2 /*return*/];
                }
            });
        });
    };
    Tensorflow1dCnnClassifier.prototype.autoDetect = function () {
        var list = Array.from(new Set(this.data.inputLabels.map(function (value) { return value[0]; })));
        list.sort(function (a, b) { return a - b; });
        return list;
    };
    Tensorflow1dCnnClassifier.prototype.setActivation = function (activation) {
        this.data.activation = activation;
    };
    Tensorflow1dCnnClassifier.prototype.setLabels = function (labels) {
        var list = Array.from(new Set(labels));
        list.sort(function (a, b) { return a - b; });
        this.data.labels = list;
    };
    Tensorflow1dCnnClassifier.prototype.setBatchSize = function (batch) {
        this.data.batchSize = batch;
    };
    Tensorflow1dCnnClassifier.prototype.getTestingInputs = function () {
        return {
            input: [
                {
                    id: 'input',
                    label: 'Testing Input Features',
                    min: 1,
                    max: 1,
                    type: 'number[]'
                }
            ],
            output: [
                {
                    id: 'output',
                    label: 'Testing Output Feature',
                    min: 1,
                    max: 1,
                    type: 'number'
                }
            ]
        };
    };
    return Tensorflow1dCnnClassifier;
}(data_science_lab_core_1.AlgorithmPlugin));
exports.Tensorflow1dCnnClassifier = Tensorflow1dCnnClassifier;
var Tensorflow1dCnnClassifierPluginOptions = /** @class */ (function (_super) {
    __extends(Tensorflow1dCnnClassifierPluginOptions, _super);
    function Tensorflow1dCnnClassifierPluginOptions(classifier) {
        var _this = _super.call(this) || this;
        _this.classifier = classifier;
        _this.state = 1;
        _this.labels = [];
        return _this;
    }
    Tensorflow1dCnnClassifierPluginOptions.prototype.submit = function (inputs) {
        switch (this.state) {
            case 1:
                this.classifier.setActivation(inputs['activation']);
                this.classifier.setBatchSize(inputs['batch']);
                this.labels = this.classifier.autoDetect();
                this.state = 2;
                break;
            case 3:
                this.classifier.setLabels(JSON.parse("[" + inputs['labels'] + "]"));
                this.state = 4;
                break;
            default:
                throw new Error("Tensorflow 1D CNN Classifion in invalid state.");
        }
    };
    Tensorflow1dCnnClassifierPluginOptions.prototype.executeCommand = function (id) {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                if (this.state === 2) {
                    if (id === 'yes') {
                        this.classifier.setLabels(this.labels);
                        this.state = 4;
                    }
                    else if (id === 'no') {
                        this.state = 3;
                    }
                    else {
                        throw new Error("Tensorflow 1D CNN Classification got invalid command: " + id);
                    }
                }
                else {
                    throw new Error("Tensorflow 1D CNN Classifion in invalid state.");
                }
                return [2 /*return*/];
            });
        });
    };
    Tensorflow1dCnnClassifierPluginOptions.prototype.options = function () {
        switch (this.state) {
            case 1:
                return [
                    new data_science_lab_core_1.ChoicesOption({
                        id: 'activation',
                        choices: [
                            'softmax',
                            'elu',
                            'hardSigmoid',
                            'relu',
                            'relu6',
                            'selu',
                            'sigmoid',
                            'softplus',
                            'softsign',
                            'tanh'
                        ],
                        label: 'Choose an activation function'
                    }),
                    new data_science_lab_core_1.NumberOption({
                        id: 'batch',
                        label: 'Choose a batch size',
                        min: 1,
                        step: 100
                    })
                ];
            case 2:
                return [
                    new data_science_lab_core_1.CommandOption({
                        id: 'yes',
                        command: 'Yes',
                        label: "Are these labels " + this.labels + " correct?",
                    }),
                    new data_science_lab_core_1.CommandOption({
                        id: 'no',
                        command: 'No',
                        label: 'Incorrect. Will go to manual input when click',
                    })
                ];
            case 3:
                return [
                    new data_science_lab_core_1.TextOption({
                        id: 'labels',
                        label: 'Input Label List. (example input: 1,2,3,4)',
                        min: 1,
                        pattern: '([ ]*[0-9]+[ ]*)(,[ ]*[0-9]+[ ]*)+'
                    })
                ];
            default:
                throw new Error("Tensorflow 1D CNN Classifier in invalid state.");
        }
    };
    Tensorflow1dCnnClassifierPluginOptions.prototype.noMore = function () {
        return this.state === 4;
    };
    return Tensorflow1dCnnClassifierPluginOptions;
}(data_science_lab_core_1.PluginOptions));
var Tensorflow1dCnnClassifierPluginInputs = /** @class */ (function (_super) {
    __extends(Tensorflow1dCnnClassifierPluginInputs, _super);
    function Tensorflow1dCnnClassifierPluginInputs(classifier) {
        var _this = _super.call(this) || this;
        _this.classifier = classifier;
        return _this;
    }
    Tensorflow1dCnnClassifierPluginInputs.prototype.inputs = function () {
        return [
            {
                id: 'input',
                label: 'Input Array Feature',
                min: 1,
                max: 1,
                type: 'number[]'
            },
            {
                id: 'output',
                label: 'Output Feature',
                min: 1,
                max: 1,
                type: 'number'
            }
        ];
    };
    Tensorflow1dCnnClassifierPluginInputs.prototype.submit = function (inputs) {
        if (inputs['input'] === undefined) {
            throw new Error("Tensorflow 1D CNN Classifion's submit expecting plugin data with key input");
        }
        else {
            this.classifier.setInput(inputs['input'].examples.map(function (value) { return value[0]; }));
        }
        if (inputs['output'] === undefined) {
            throw new Error("Tensorflow 1D CNN Classifion's submit expecting plugin data with key output");
        }
        else {
            this.classifier.setOutput(inputs['output'].examples);
        }
    };
    return Tensorflow1dCnnClassifierPluginInputs;
}(data_science_lab_core_1.PluginInputs));
