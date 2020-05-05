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
var tensorflow_1d_cnn_classifier_1 = require("./tensorflow-1d-cnn.classifier");
describe("Tesnorflow 1d CNN Classifier Tests", function () {
    var classifier;
    /*
        Number 1
        5 | 0 1 1 0 0
        4 | 0 0 1 0 0
        3 | 0 0 1 0 0
        2 | 0 0 1 0 0
        1 | 1 1 1 1 1
        -------------
        #   1 2 3 4 5
        5 | 1 1 0 0 0
        4 | 0 1 0 0 0
        3 | 0 1 0 0 0
        2 | 0 1 0 0 0
        1 | 1 1 1 1 0
        -------------
        #   1 2 3 4 5
        5 | 0 0 1 1 0
        4 | 0 0 0 1 0
        3 | 0 0 0 1 0
        2 | 0 0 0 1 0
        1 | 1 1 1 1 1
        -------------
        #   1 2 3 4 5
        
        Number 2
        5 | 1 1 1 1 1
        4 | 0 0 0 1 0
        3 | 0 0 1 0 0
        2 | 0 1 0 0 0
        1 | 1 1 1 1 1
        -------------
        #   1 2 3 4 5
        5 | 1 1 1 1 1
        4 | 0 0 1 1 0
        3 | 0 0 1 0 0
        2 | 0 1 1 0 0
        1 | 1 1 1 1 1
        -------------
        #   1 2 3 4 5
        5 | 1 1 1 1 1
        4 | 0 0 0 1 1
        3 | 0 0 1 1 0
        2 | 0 1 0 0 0
        1 | 1 1 1 1 1
        -------------
        #   1 2 3 4 5
        Number 3
        5 | 1 1 1 1 1
        4 | 0 0 0 0 1
        3 | 0 0 1 1 1
        2 | 0 0 0 0 1
        1 | 1 1 1 1 1
        -------------
        #   1 2 3 4 5
        5 | 1 1 1 1 1
        4 | 0 0 0 0 1
        3 | 0 0 1 1 1
        2 | 0 0 0 0 1
        1 | 1 1 1 1 1
        -------------
        #   1 2 3 4 5
        5 | 1 1 1 1 1
        4 | 0 0 0 0 1
        3 | 0 0 1 1 1
        2 | 0 0 0 0 1
        1 | 1 1 1 1 1
        -------------
        #   1 2 3 4 5
*/
    var testingInput = {
        'input': {
            features: ['picture'],
            examples: [
                [[0, 1, 1, 0, 0,
                        0, 0, 1, 0, 0,
                        0, 0, 1, 0, 0,
                        0, 0, 1, 0, 0,
                        1, 1, 1, 1, 1]],
                [[1, 1, 0, 0, 0,
                        0, 1, 0, 0, 0,
                        0, 1, 0, 0, 0,
                        0, 1, 0, 0, 0,
                        1, 1, 1, 1, 0]],
                [[0, 0, 1, 1, 0,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 1, 0,
                        1, 1, 1, 1, 1]],
                [[1, 1, 1, 1, 1,
                        0, 0, 0, 1, 0,
                        0, 0, 1, 0, 0,
                        0, 1, 0, 0, 0,
                        1, 1, 1, 1, 1]],
                [[1, 1, 1, 1, 1,
                        0, 0, 1, 1, 0,
                        0, 0, 1, 0, 0,
                        0, 1, 1, 0, 0,
                        1, 1, 1, 1, 1]],
                [[1, 1, 1, 1, 1,
                        0, 0, 0, 1, 1,
                        0, 0, 1, 1, 0,
                        0, 1, 0, 0, 0,
                        1, 1, 1, 1, 1]],
                [[1, 1, 1, 1, 1,
                        0, 0, 0, 0, 1,
                        0, 0, 1, 1, 1,
                        0, 0, 0, 0, 1,
                        1, 1, 1, 1, 1]],
                [[1, 1, 1, 1, 1,
                        0, 0, 0, 0, 1,
                        0, 0, 1, 1, 1,
                        0, 0, 0, 0, 1,
                        1, 1, 1, 1, 1]],
                [[1, 1, 1, 1, 1,
                        0, 0, 0, 0, 1,
                        0, 0, 1, 1, 1,
                        0, 0, 0, 0, 1,
                        1, 1, 1, 1, 1]]
            ]
        },
        'output': {
            features: ['label'],
            examples: [
                [1],
                [1],
                [1],
                [2],
                [2],
                [2],
                [3],
                [3],
                [3],
            ]
        }
    };
    beforeEach(function () {
        classifier = new tensorflow_1d_cnn_classifier_1.Tensorflow1dCnnClassifier();
    });
    it('get inputs should return two', function () {
        var inputs = classifier.getInputs().inputs();
        expect(inputs.length).toBe(2);
    });
    it('submit should throw throw for no input', function () {
        expect(function () {
            classifier.getInputs().submit({
                'output': {
                    features: ['label'],
                    examples: [[1]]
                }
            });
        }).toThrowError();
    });
    it('submit should throw throw for no output', function () {
        expect(function () {
            classifier.getInputs().submit({
                'output': {
                    features: ['label'],
                    examples: [[1]]
                }
            });
        }).toThrowError();
    });
    describe('after submit input', function () {
        beforeEach(function () {
            classifier.getInputs().submit(JSON.parse(JSON.stringify(testingInput)));
        });
        it('get options should return false for noMore', function () {
            expect(classifier.getOptions().noMore()).toBeFalsy();
        });
        it('get options should return two options', function () {
            expect(classifier.getOptions().options().length).toBe(2);
        });
        it('get option submit match and activation should return false for noMore', function () {
            classifier.getOptions().submit({
                'activation': 'softmax',
                'batch': 5
            });
            expect(classifier.getOptions().noMore()).toBeFalsy();
        });
        it('get options should prompt yes and no after first options and no more when yes', function () { return __awaiter(void 0, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        classifier.getOptions().submit({
                            'activation': 'softmax',
                            'batch': 5
                        });
                        return [4 /*yield*/, classifier.getOptions().executeCommand('yes')];
                    case 1:
                        _a.sent();
                        expect(classifier.getOptions().noMore()).toBeTruthy();
                        expect(classifier.data.labels).toEqual([1, 2, 3]);
                        return [2 /*return*/];
                }
            });
        }); });
        it('get options should prompt yes and no after first options and false no more when no', function () { return __awaiter(void 0, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        classifier.getOptions().submit({
                            'activation': 'softmax',
                            'batch': 5
                        });
                        return [4 /*yield*/, classifier.getOptions().executeCommand('no')];
                    case 1:
                        _a.sent();
                        expect(classifier.getOptions().noMore()).toBeFalsy();
                        expect(classifier.data.batchSize).toBe(5);
                        expect(classifier.data.activation).toBe('softmax');
                        expect(classifier.getOptions().options().length).toBe(1);
                        return [2 /*return*/];
                }
            });
        }); });
        it('get options should prompt yes and no after first options and submitted labels', function () { return __awaiter(void 0, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        classifier.getOptions().submit({
                            'activation': 'softmax',
                            'batch': 5
                        });
                        return [4 /*yield*/, classifier.getOptions().executeCommand('no')];
                    case 1:
                        _a.sent();
                        classifier.getOptions().submit({
                            'labels': '1,2,3',
                        });
                        expect(classifier.getOptions().noMore()).toBeTruthy();
                        expect(classifier.data.labels).toEqual([1, 2, 3]);
                        return [2 /*return*/];
                }
            });
        }); });
        describe('after options', function () {
            beforeEach(function (done) { return __awaiter(void 0, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            classifier.getOptions().submit({
                                'activation': 'softmax',
                                'batch': 5
                            });
                            return [4 /*yield*/, classifier.getOptions().executeCommand('yes')];
                        case 1:
                            _a.sent();
                            classifier.initialize();
                            done();
                            return [2 /*return*/];
                    }
                });
            }); });
            it('expect finish training to be false', function () {
                expect(classifier.finishTraining()).toBeFalsy();
            });
            it('get testing input should return one for each', function () {
                var testing = classifier.getTestingInputs();
                expect(testing.input.length).toBe(1);
                expect(testing.output).toBeDefined();
                if (testing.output) {
                    expect(testing.output.length).toBe(1);
                }
            });
            it('trainX to have shape of 9,25,1 ', function () {
                expect(classifier.data.trainX.shape).toEqual([9, 25, 1]);
            });
            it('trainLabels to have shape of 9,3 ', function () {
                expect(classifier.data.trainLabels.shape).toEqual([9, 3]);
            });
            it('set recorded one step should call recorder', function () { return __awaiter(void 0, void 0, void 0, function () {
                var recorder;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            recorder = jasmine.createSpyObj('RecorderService', ['record']);
                            classifier.setRecorderService(recorder);
                            return [4 /*yield*/, classifier.step()];
                        case 1:
                            _a.sent();
                            expect(recorder.record).toHaveBeenCalledTimes(1);
                            return [2 /*return*/];
                    }
                });
            }); });
            it('training few steps should be able to predict the training set', function () { return __awaiter(void 0, void 0, void 0, function () {
                var i, i, actual;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            i = 0;
                            _a.label = 1;
                        case 1:
                            if (!(i < 110)) return [3 /*break*/, 4];
                            return [4 /*yield*/, classifier.step()];
                        case 2:
                            _a.sent();
                            _a.label = 3;
                        case 3:
                            ++i;
                            return [3 /*break*/, 1];
                        case 4:
                            for (i = 0; i < testingInput.output.examples.length; ++i) {
                                actual = classifier.test({ 'input': testingInput.input.examples[i] });
                                expect(actual.output).toEqual(testingInput.output.examples[i]);
                            }
                            return [2 /*return*/];
                    }
                });
            }); });
            it('export and import without minimial should be able to train', function () { return __awaiter(void 0, void 0, void 0, function () {
                var json, newClassifier, i, i, actual;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4 /*yield*/, classifier.export(false)];
                        case 1:
                            json = _a.sent();
                            return [4 /*yield*/, (new tensorflow_1d_cnn_classifier_1.Tensorflow1dCnnClassifier()).import(json, false)];
                        case 2:
                            newClassifier = _a.sent();
                            i = 0;
                            _a.label = 3;
                        case 3:
                            if (!(i < 100)) return [3 /*break*/, 6];
                            return [4 /*yield*/, newClassifier.step()];
                        case 4:
                            _a.sent();
                            _a.label = 5;
                        case 5:
                            ++i;
                            return [3 /*break*/, 3];
                        case 6:
                            for (i = 0; i < testingInput.output.examples.length; ++i) {
                                actual = newClassifier.test({ 'input': testingInput.input.examples[i] });
                                expect(actual.output).toEqual(testingInput.output.examples[i]);
                            }
                            return [2 /*return*/];
                    }
                });
            }); });
            it('train most export and import without minimial should be able to train rest', function () { return __awaiter(void 0, void 0, void 0, function () {
                var i, json, newClassifier, i, i, actual;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            i = 0;
                            _a.label = 1;
                        case 1:
                            if (!(i < 100)) return [3 /*break*/, 4];
                            return [4 /*yield*/, classifier.step()];
                        case 2:
                            _a.sent();
                            _a.label = 3;
                        case 3:
                            ++i;
                            return [3 /*break*/, 1];
                        case 4: return [4 /*yield*/, classifier.export(false)];
                        case 5:
                            json = _a.sent();
                            return [4 /*yield*/, (new tensorflow_1d_cnn_classifier_1.Tensorflow1dCnnClassifier()).import(json, false)];
                        case 6:
                            newClassifier = _a.sent();
                            i = 0;
                            _a.label = 7;
                        case 7:
                            if (!(i < 10)) return [3 /*break*/, 10];
                            return [4 /*yield*/, newClassifier.step()];
                        case 8:
                            _a.sent();
                            _a.label = 9;
                        case 9:
                            ++i;
                            return [3 /*break*/, 7];
                        case 10:
                            for (i = 0; i < testingInput.output.examples.length; ++i) {
                                actual = newClassifier.test({ 'input': testingInput.input.examples[i] });
                                expect(actual.output).toEqual(testingInput.output.examples[i]);
                            }
                            return [2 /*return*/];
                    }
                });
            }); });
            it('testing after export minimal should still work', function () { return __awaiter(void 0, void 0, void 0, function () {
                var i, json, newClassifier, i, actual;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            i = 0;
                            _a.label = 1;
                        case 1:
                            if (!(i < 110)) return [3 /*break*/, 4];
                            return [4 /*yield*/, classifier.step()];
                        case 2:
                            _a.sent();
                            _a.label = 3;
                        case 3:
                            ++i;
                            return [3 /*break*/, 1];
                        case 4: return [4 /*yield*/, classifier.export(true)];
                        case 5:
                            json = _a.sent();
                            return [4 /*yield*/, (new tensorflow_1d_cnn_classifier_1.Tensorflow1dCnnClassifier()).import(json, true)];
                        case 6:
                            newClassifier = _a.sent();
                            for (i = 0; i < testingInput.output.examples.length; ++i) {
                                actual = newClassifier.test({ 'input': testingInput.input.examples[i] });
                                expect(actual.output).toEqual(testingInput.output.examples[i]);
                            }
                            return [2 /*return*/];
                    }
                });
            }); });
        });
    });
});
