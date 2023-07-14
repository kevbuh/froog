# JSML

js | ml

crazy? js for machine learning and ai???

maybe but i think it would be fun to create a machine learning framework with the best parts of tinygrad and tensorflow js and pytorch. 

Why Javascript?
JS for backend, for database, for frontend, for mobile apps, for desktop apps, and yes, for AI. One language to rule them all.

- First of all, JavaScript is the most commonly used programming language on the planet, and you can do anything in js
- JavaScript enables zero-download demonstrations that will allow researchers to share their models and results more easily with a large audience. This will be important as ai/ml starts getting integrated into everday life with everyday people
- JavaScript runs client-side, which will reduce the cost associated with hosting model-running servers
- Opens new horizons for apps that you can build.  Rather than being limited to deploying Python code on the server for running your ML code, you can build single-page apps, or even browser extensions that run interesting algorithms, which can give you the possibility of developing a completely novel use case!
- Asynchronous programming. This opens the door for many types of backend implementations (including those that either JIT compile or download operator implementations on the fly).
- In-place operations

Summary
JS lots of graphing libraries, ease of use, access to TensorFlow, and portability. 

NPM
npm is a pain. lets just make everything we need from scratch——apple style. scientific/numbers/data libraries, the whole package

Speed
- Despite a JS frontend, implementing compute-heavy operators in pure JavaScript is likely insufficient for general usability.
- WebAssembly for nearly 100% coverage (all major browsers support it)
- WebGPU - Much faster than wasm but lower coverage
- Will try webGPU and default to WebAsm

Interoperability with Python
- optional ability to author / train in python and export their model to be loaded in javascript for demo

Language
```javascript
export class Net extends nn.Module {
  constructor() {
    super();
    this.convs = nn.Sequential(
      nn.Conv2d(3, 16, 5, {stride: 1, padding: 2}),
      nn.MaxPool2d(2),
      nn.ReLU(),
      nn.Conv2d(16, 32, 5, {stride: 1, padding: 2}),
      nn.MaxPool2d(2),
      nn.ReLU()
    );
    this.mlp = nn.Linear(32, 10);
  }
  
  forward(input) {
    let x = input;
    x = this.convs(x);
    x = x.view([x.size(0), -1]);
    return this.mlp(x);
  }
}

const model = new Net();
await model.load_state_dict("model.ptjs");
```

TypeScript? 
Maybe need to look into it more. Could add typescript for static type checking.

Sources
- https://dev-discuss.pytorch.org/t/proposal-torch-js-a-javascript-frontend-for-pytorch/650