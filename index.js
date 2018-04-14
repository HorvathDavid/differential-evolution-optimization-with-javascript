import _ from 'lodash'
import * as tf from '@tensorflow/tfjs';

window.tf = tf

const inputText = `long ago , the mice had a general council to consider what measures they could take to outwit their common enemy , the cat . some said this , and some said that but at last a young mouse got up and said he had a proposal to make , which he thought would meet the case . you will all agree , said he , that our chief danger consists in the sly and treacherous manner in which the enemy approaches us . now , if we could receive some signal of her approach , we could easily escape from her . i venture , therefore , to propose that a small bell be procured , and attached by a ribbon round the neck of the cat . by this means we should always know when she was about , and could easily retire while she was in the neighbourhood . this proposal met with general applause , until an old mouse got up and said that is all very well , but who is to bell the cat ? the mice looked at one another and nobody spoke . then the old mouse said it is easy to propose impossible remedies .`

const numIterations = 10000
const learning_rate = 0.001
const rnn_hidden = 100
const preparedDataforTestSet = inputText.split(' ')
const endOfSeq = preparedDataforTestSet.length - 4
const optimizer = tf.train.rmsprop(learning_rate)


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

const wordMap = createWordMap(inputText)
const wordWrapLength = Object.keys(wordMap).length

// console.log(wordMap)


// return a word
const fromSymbol = (symbol) => {
    const object = wordMap.filter(e => e.code === symbol)[0]
    return object.word
}

// return a symbol
const toSymbol = (word) => {
    const object = wordMap.filter(e => e.word === word)[0]
    return object.code
}

// return onehot vector, for compare with probability distribution vector
const encode = (symbol) => {
    // console.log(symbol)
    return tf.oneHot(tf.tensor1d(symbol), wordWrapLength)
}

// return a symbol
const decode = (probDistVector) => {

    // @todo: ide kell majd beleirni
    const some = probDistVector.dataSync()
    const a = some.indexOf(_.max(some))
    console.log(a)
    return a
}


// building the model
const wordVector = tf.input({ shape: [3, 1] });
const cells = [
    tf.layers.lstmCell({ units: rnn_hidden }),
    // tf.layers.lstmCell({ units: rnn_hidden }),
];
const rnn = tf.layers.rnn({ cell: cells, returnSequences: false });

const rnn_out = rnn.apply(wordVector);
const output = tf.layers.dense({ units: wordWrapLength, activation: 'softmax', useBias: true }).apply(rnn_out)

const model = tf.model({ inputs: wordVector, outputs: output })


// sample is: shape: [batch, sequence, feature], here is [1, number of wordsq, 1]
const predict = (samples) => {
    // console.log(samples)
    return tf.tidy(() => {
        return model.predict(samples)
    })
}

const loss = (labels, predictions) => {
    // console.log(labels, predictions)
    return tf.losses.softmaxCrossEntropy(labels, predictions).mean();
}

// performance could be improved if toSymbol the whole set
// then random select from encodings not from string of arrays
const getSamples = () => {
    const startOfSeq = _.random(0, endOfSeq, false)
    const retVal = preparedDataforTestSet.slice(startOfSeq, startOfSeq + 4)
    return retVal
}

const train = async (numIterations) => {
    for (let iter = 0; iter < numIterations; iter++) {

        const samples = getSamples().map(s => {
            return toSymbol(s)
        })

        const labelProbVector = encode(samples.splice(-1))

        // optimizer.minimize is where the training happens. 

        // The function it takes must return a numerical estimate (i.e. loss) 
        // of how well we are doing using the current state of
        // the variables we created at the start.

        // This optimizer does the 'backward' step of our training process
        // updating variables defined previously in order to minimize the
        // loss.
        const lossValue = optimizer.minimize(() => {
            // Feed the examples into the model
            const pred = predict(tf.tensor(samples, [1, 3, 1]));
            return loss(labelProbVector, pred);
        }, true);
        
        console.log(`The loss is:  ${lossValue.dataSync()}   --------`)
        // console.log(lossValue)
        // Use tf.nextFrame to not block the browser.
        await tf.nextFrame();
    }
}

const learnToGuessWord = async () => {
    console.log('TRAIN START')
    await train(numIterations);

    console.log('TRAIN IS OVER')

    const symbolCollector = getSamples().map(s => {
        return toSymbol(s)
    })

    for (let i = 0; i < 10; i++) {
        const predProbVector = predict(tf.tensor(symbolCollector.slice(-3), [1, 3, 1]))
        symbolCollector.push(decode(predProbVector));
    }

    const generatedText = symbolCollector.map(s => {
        return fromSymbol(s)
    }).join(' ')

    console.log(generatedText)
}

learnToGuessWord();
// // TEST COMES HERE
// const tensorTest = getSamples().map(s => {
//     return encode(s)
// }).slice(0, 3)

// // console.log(tensorTest)
// const shit = tf.tensor([32, 46, 52], [1, 3, 1])
// shit.print()

// predict(shit).print()