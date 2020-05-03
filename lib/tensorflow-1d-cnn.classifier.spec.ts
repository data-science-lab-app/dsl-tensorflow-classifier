import { Tensorflow1dCnnClassifier } from './tensorflow-1d-cnn.classifier';

describe("Tesnorflow 1d CNN Classifier Tests", () => {
    let classifier: Tensorflow1dCnnClassifier;

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
    const testingInput = {
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
    }

    beforeEach(() => {
        classifier = new Tensorflow1dCnnClassifier();
    });

    it('get inputs should return two', () => {
        const inputs = classifier.getInputs().inputs();
        expect(inputs.length).toBe(2);
    });

    it('submit should throw throw for no input', () => {
        expect(() => {
            classifier.getInputs().submit({
                'output': {
                    features: ['label'],
                    examples: [[1]]
                }
            });
        }).toThrowError();
    });


    it('submit should throw throw for no output', () => {
        expect(() => {
            classifier.getInputs().submit({
                'output': {
                    features: ['label'],
                    examples: [[1]]
                }
            });
        }).toThrowError();
    });

    describe('after submit input', () => {
        beforeEach(() => {
            classifier.getInputs().submit(testingInput);
        });

        it('get options should return false for noMore', () => {
            expect(classifier.getOptions().noMore()).toBeFalsy();
        });

        it('get options should return two options', () => {
            expect(classifier.getOptions().options().length).toBe(2);
        });

        it('get option submit match and activation should return false for noMore', () => {
            classifier.getOptions().submit({
                'activation': 'softmax',
                'batch': 5
            });
            expect(classifier.getOptions().noMore()).toBeFalsy();
        });

        it('get options should prompt yes and no after first options and no more when yes', async () => {
            classifier.getOptions().submit({
                'activation': 'softmax',
                'batch': 5
            });
            await classifier.getOptions().executeCommand('yes');
            expect(classifier.getOptions().noMore()).toBeTruthy();
            expect(classifier.data.labels).toEqual([1, 2, 3]);
        });


        it('get options should prompt yes and no after first options and false no more when no', async () => {
            classifier.getOptions().submit({
                'activation': 'softmax',
                'batch': 5
            });
            await classifier.getOptions().executeCommand('no');
            expect(classifier.getOptions().noMore()).toBeFalsy();
            expect(classifier.data.batchSize).toBe(5);
            expect(classifier.data.activation).toBe('softmax');
            expect(classifier.getOptions().options().length).toBe(1);
        });

        it('get options should prompt yes and no after first options and submitted labels', async () => {
            classifier.getOptions().submit({
                'activation': 'softmax',
                'batch': 5
            });
            await classifier.getOptions().executeCommand('no');
            classifier.getOptions().submit({
                'labels': '1,2,3',
            });
            expect(classifier.getOptions().noMore()).toBeTruthy();
            expect(classifier.data.labels).toEqual([1, 2, 3]);
        });

        describe('after options', () => {
            beforeEach(async (done) => {
                classifier.getOptions().submit({
                    'activation': 'softmax',
                    'batch': 5
                });
                await classifier.getOptions().executeCommand('yes');
                classifier.initialize();
                done();
            });

            it('expect finish training to be false', () => {
                expect(classifier.finishTraining()).toBeFalsy();
            })

            it('get testing input should return one for each', () => {
                const testing = classifier.getTestingInputs();
                expect(testing.input.length).toBe(1);
                expect(testing.output).toBeDefined();
                if (testing.output) {
                    expect(testing.output.length).toBe(1);
                }
            });

            it('trainX to have shape of 9,25,1 ', () => {
                expect(classifier.data.trainX.shape).toEqual([9, 25, 1]);
            });

            it('trainLabels to have shape of 9,3 ', () => {
                expect(classifier.data.trainLabels.shape).toEqual([9, 3]);
            });

            it('set recorded one step should call recorder', async () => {
                const recorder = jasmine.createSpyObj('RecorderService', ['record']);
                classifier.setRecorderService(recorder);
                await classifier.step();
                expect(recorder.record).toHaveBeenCalledTimes(1);
            });

            it('training few steps should be able to predict the training set', async () => {
                for (let i = 0; i < 1000; ++i) {
                    await classifier.step();
                }
                for (let i = 0; i < testingInput.output.examples.length; ++i) {
                    const actual = classifier.test(
                        { 'input': testingInput.input.examples[i] }
                    );
                    expect(actual.output).toEqual(testingInput.output.examples[i]);
                }
            });

            it('export and import without minimial should be able to train', async () => {
                const json = await classifier.export(false);
                let newClassifier = await (new Tensorflow1dCnnClassifier()).import(json, false);
                for (let i = 0; i < 1000; ++i) {
                    await newClassifier.step();
                }
                for (let i = 0; i < testingInput.output.examples.length; ++i) {
                    const actual = newClassifier.test(
                        { 'input': testingInput.input.examples[i] }
                    );
                    expect(actual.output).toEqual(testingInput.output.examples[i]);
                }
            });
    
            it('testing after export minimal should still work', async () => {
                for (let i = 0; i < 1000; ++i) {
                    await classifier.step();
                } 
                const json = await classifier.export(true);
                let newClassifier = await (new Tensorflow1dCnnClassifier()).import(json, true);
                for (let i = 0; i < testingInput.output.examples.length; ++i) {
                    const actual = newClassifier.test(
                        { 'input': testingInput.input.examples[i] }
                    );
                    expect(actual.output).toEqual(testingInput.output.examples[i]);
                }
            });

        });
    });
});