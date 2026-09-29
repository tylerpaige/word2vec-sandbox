const express = require("express");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.static(path.join(__dirname, "webapp")));

app.listen(PORT, () => {
  console.log(`Word Algebra preview: http://localhost:${PORT}`);
});
