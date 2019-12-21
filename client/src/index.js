import React from 'react';
import { render } from 'react-dom';
import { AppContainer } from 'react-hot-loader';
import { Provider } from 'react-redux'
import { applyMiddleware, createStore, combineReducers, compose } from 'redux'
import thunk from 'redux-thunk'
import App from './app.js'
import reducers from './reducer'
import * as serviceWorker from './serviceWorker'

const reducer = combineReducers(reducers)
const store = createStore(
  reducer,
  compose(applyMiddleware(thunk), window.devToolsExtension ? window.devToolsExtension() : f => f)
)

render( <AppContainer><Provider store={store}><App/></Provider></AppContainer>, document.getElementById("root"));

if (module && module.hot) {
  module.hot.accept('./app.js', () => {
    const App = require('./app.js').default;
    render(
      <AppContainer>
        <Provider store={store}><App/></Provider>
      </AppContainer>,
      document.getElementById("app")  
    );
  });
}


// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();