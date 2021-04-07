// Pull in native module that we have no information about
const basis = require('./build/basis.node')


// Create thin wrappers that give Node context and WebStorm autocomplete

/**
 * A test function that just returns a pre-defined message.
 *
 * @returns {string}
 */
function greeting() {
    return basis['greeting']()
}


/**
 * A test function that adds two numbers
 *
 * @param {number} x
 * @param {number} y
 *
 * @returns {number}
 */
function add(x, y) {
    return basis['add'](x, y)
}


module.exports = {greeting, add}