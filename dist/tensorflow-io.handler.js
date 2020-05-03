"use strict";
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
var TensorflowIOHandler = /** @class */ (function () {
    function TensorflowIOHandler(json) {
        this.json = json;
    }
    Object.defineProperty(TensorflowIOHandler.prototype, "save", {
        get: function () {
            var _this = this;
            return function (modelArtifact) {
                return new Promise(function (resolve) { return __awaiter(_this, void 0, void 0, function () {
                    var data, weightData, _a;
                    return __generator(this, function (_b) {
                        switch (_b.label) {
                            case 0:
                                data = {
                                    artifact: modelArtifact,
                                };
                                weightData = modelArtifact.weightData;
                                if (!weightData) return [3 /*break*/, 2];
                                _a = data;
                                return [4 /*yield*/, this.saveWeights(weightData)];
                            case 1:
                                _a.weights = _b.sent();
                                data.artifact.weightData = undefined;
                                _b.label = 2;
                            case 2:
                                this.json = JSON.stringify(data);
                                resolve({
                                    modelArtifactsInfo: {
                                        dateSaved: new Date(),
                                        modelTopologyType: 'JSON',
                                        modelTopologyBytes: modelArtifact.modelTopology == null ? 0 : Buffer.byteLength(JSON.stringify(modelArtifact.modelTopology), 'utf-8'),
                                        weightSpecsBytes: modelArtifact.weightSpecs == null ? 0 : Buffer.byteLength(JSON.stringify(modelArtifact.weightSpecs), 'utf-8'),
                                        weightDataBytes: weightData == null ? 0 : weightData.byteLength
                                    }
                                });
                                return [2 /*return*/];
                        }
                    });
                }); });
            };
        },
        enumerable: true,
        configurable: true
    });
    ;
    Object.defineProperty(TensorflowIOHandler.prototype, "load", {
        get: function () {
            var _this = this;
            return function () {
                return new Promise(function (resolve, reject) { return __awaiter(_this, void 0, void 0, function () {
                    var modelJson, modelArtifacts, _a;
                    return __generator(this, function (_b) {
                        switch (_b.label) {
                            case 0:
                                if (!this.json) return [3 /*break*/, 3];
                                modelJson = JSON.parse(this.json);
                                modelArtifacts = {
                                    modelTopology: modelJson.artifact.modelTopology,
                                    format: modelJson.artifact.format,
                                    generatedBy: modelJson.artifact.generatedBy,
                                    convertedBy: modelJson.artifact.convertedBy,
                                    trainingConfig: modelJson.artifact.trainingConfig,
                                    userDefinedMetadata: modelJson.artifact.userDefinedMetadata
                                };
                                if (!(modelJson.weights !== undefined)) return [3 /*break*/, 2];
                                _a = modelArtifacts;
                                return [4 /*yield*/, this.loadWeights(modelJson.weights)];
                            case 1:
                                _a.weightData = _b.sent();
                                modelArtifacts.weightSpecs = modelJson.artifact.weightSpecs;
                                _b.label = 2;
                            case 2:
                                resolve(modelArtifacts);
                                return [3 /*break*/, 4];
                            case 3:
                                reject(new Error("Load requires a json object!"));
                                _b.label = 4;
                            case 4: return [2 /*return*/];
                        }
                    });
                }); });
            };
        },
        enumerable: true,
        configurable: true
    });
    TensorflowIOHandler.prototype.saveWeights = function (weights) {
        return __awaiter(this, void 0, void 0, function () {
            var buffer;
            return __generator(this, function (_a) {
                buffer = Buffer.from(weights);
                return [2 /*return*/, JSON.stringify(buffer.toJSON())];
            });
        });
    };
    TensorflowIOHandler.prototype.loadWeights = function (weights) {
        return __awaiter(this, void 0, void 0, function () {
            var buffer, arrayBuffer;
            return __generator(this, function (_a) {
                buffer = Buffer.from(JSON.parse(weights));
                arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
                return [2 /*return*/, arrayBuffer];
            });
        });
    };
    return TensorflowIOHandler;
}());
exports.TensorflowIOHandler = TensorflowIOHandler;