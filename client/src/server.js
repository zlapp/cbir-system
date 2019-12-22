const path = require('path')
const express = require('express')
const cors = require('cors')

module.exports = {
  app: function () {    
    var app = express();
    app.use(cors());
    const indexPath = path.join(__dirname, 'index.html');
    const publicPath = express.static(path.join(__dirname, '../public'));

    app.use('/public', publicPath);
    app.get('/', function (_, res) { res.sendFile(indexPath) });

    return app;
  }
}
