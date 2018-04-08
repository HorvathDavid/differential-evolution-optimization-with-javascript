# differential-evolution-optimization-with-javascript
Differential evolution optimization in the browser. Based on this great article:

> https://nathanrooy.github.io/posts/2017-08-27/simple-differential-evolution-with-python/

The code is a rewrite in JavaScript. Every little details belongs to the original developer. See the original repository:

> https://github.com/nathanrooy/differential-evolution-optimization-with-python

## For test run
1. Clone repository
2. Open index.html in broswser. The already bundled js will load, so the cost function could not be changed. 
## For development
- Install local dependencies
    
    `npm install`
- Install webpack globally
    
    `npm install webpack webpack-cli -g`
- Install http-server for loading assets, etc.

    `npm install http-server -g`

Finally run:
- npm run webpack
- http-server

in separete terminals.

The new tensorflow.js also included in the project.
Experiment with it!
> https://js.tensorflow.org/

