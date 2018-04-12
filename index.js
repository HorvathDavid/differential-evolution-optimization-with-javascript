import _ from 'lodash'
import * as tf from '@tensorflow/tfjs';

window.tf = tf

const inputText = `long ago , the mice had a general council to consider what measures they could take to outwit their common enemy , the cat . some said this , and some said that but at last a young mouse got up and said he had a proposal to make , which he thought would meet the case . you will all agree , said he , that our chief danger consists in the sly and treacherous manner in which the enemy approaches us . now , if we could receive some signal of her approach , we could easily escape from her . i venture , therefore , to propose that a small bell be procured , and attached by a ribbon round the neck of the cat . by this means we should always know when she was about , and could easily retire while she was in the neighbourhood . this proposal met with general applause , until an old mouse got up and said that is all very well , but who is to bell the cat ? the mice looked at one another and nobody spoke . then the old mouse said it is easy to propose impossible remedies .`
const preparedDataforTestSet = inputText.split(' ')
const endOfSeq = preparedDataforTestSet.length - 4


// preparing data
const createWordMap = (textData) => {
    const wordArray = textData.split(' ')
    const countedWordObject = wordArray.reduce((acc, cur, i) => {
        // console.log(acc[cur])
        if (acc[cur] === undefined) {
            acc[cur] = 1
        } else {
            acc[cur] += 1
        }
        return acc
    }, {})

    const arraOfshit = []
    for (let key in countedWordObject) {
        arraOfshit.push({ word: key, occurence: countedWordObject[key] })
    }

    const wordMap = _.sortBy(arraOfshit, 'occurence').reverse().map((e, i) => {
        e['code'] = i
        return e
    })

    return wordMap
}

const decode = (prediction) => {
    return 'word'
}

const encode = (word) => {
    return 1
}

const wordMap = createWordMap(inputText)


const numIterations = 10
const learning_rate = 0.001
const optimizer = tf.RMSPropOptimizer(learning_rate)


// building the model
const biases = {
    out: tf.variable(tf.tensor([Object.keys(wordMap).length]))
}
const wordVector = tf.input({ shape: [3, 1] });
const cells = [
    tf.layers.lstmCell({ units: 4 }),
    tf.layers.lstmCell({ units: 4 }),
];
const rnn = tf.layers.rnn({ cell: cells, returnSequences: true });
const output = rnn.apply(wordVector);
const model = tf.model({ inputs: wordVector, outputs: output })
console.log(JSON.stringify(output.shape));


// sample is: shape: [batch, sequence, feature], here is [1, number of wordsq, 1]
const predict = (sample) => {
    return model.predict(sample)
}

const loss = (labels, predictions) => {
    return tf.losses.softmaxCrossEntropy(labels, predictions).mean();
}

// performance could be improved if encode the whole set
// then random select from encodings not from string of arrays
const getSamples = () => {
    const startOfSeq = _.random(0, endOfSeq, false)
    return preparedDataforTestSet.slice(startOfSeq, startOfSeq + 4)
}

const train = async (numIterations) => {
    for (let iter = 0; iter < numIterations; iter++) {

        const samples = getSamples().map(s => {
            return encode(s)
        })

        const labelSymbol =  samples.splice(-1)

        // optimizer.minimize is where the training happens. 

        // The function it takes must return a numerical estimate (i.e. loss) 
        // of how well we are doing using the current state of
        // the variables we created at the start.

        // This optimizer does the 'backward' step of our training process
        // updating variables defined previously in order to minimize the
        // loss.
        optimizer.minimize(() => {
            // Feed the examples into the model
            const pred = predict(tf.tensor(samples, [1, 3, 1]));
            return loss(pred, labelSymbol);
        });

        // Use tf.nextFrame to not block the browser.
        await tf.nextFrame();
    }
}

const learnToGuessWord = async () => {

    await train(numIterations);

    const predictedWords = getSamples().slice(-3)

    for (let i = 0; i < 3; i++) {
        predictedWords.push(decode(predict(predictedWords.slice(-3))));
    }

    console.log(predictedWords.join(' '))
}

// learnToGuessWord();


// TEST COMES HERE
// const tensorTest = getSamples().map(s => {
//     return encode(s)
// }).slice(0, 3)

// predict(tf.tensor(tensorTest, [1, 3, 1])).print()