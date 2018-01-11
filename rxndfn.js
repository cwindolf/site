/* If you like this, you will love
 * https://mrob.com/pub/comp/xmorphia/pearson-classes.html
 * which is an amazing site.
 * Do what you like with this code.
 */

class GrayScottRxnDfn {

    constructor(canvas) {
        // Load up canvas, set options, init drawing buffers.
        this.width = canvas.width;
        this.height = canvas.height;
        this.context = canvas.getContext('2d', { alpha: false });
        this.context.alphaEnabled = false;
        this.context.mozImageSmoothingEnabled = false;
        this.context.webkitImageSmoothingEnabled = false;
        this.context.msImageSmoothingEnabled = false;
        this.context.imageSmoothingEnabled = false;
        this.id = this.context.createImageData(this.width, this.height);
        this.buffer = this.id.data;

        // Diffusion rates
        this.D_u = 1.0;
        this.D_v = 0.5;

        // Gray-Scott reaction parameters.
        // We will flip between a bunch of interesting settings for fun.
        this.fs = [
            0.0141,
            0.014,
            0.022,
            0.019,
            0.026,
            0.022,
            0.026,
            0.038,
            0.042,
            0.058,
            0.062,
            0.0638,
            0.058,
            0.038,
        ]
        this.ks = [
            0.0525,
            0.05,
            0.051,
            0.0548,
            0.054,
            0.051,
            0.0565,
            0.061,
            0.059,
            0.062,
            0.061,
            0.061,
            0.062,
            0.061,
        ]

        // How simulation will move through the phases
        this.num_phases = this.fs.length;
        this.phase_freq = 1 / this.num_phases;
        this.PERIOD = 150000;

        // Buffers
        this.U = new Float64Array(this.width * this.height);
        this.U_lap = new Float64Array(this.width * this.height);
        this.V = new Float64Array(this.width * this.height);
        this.V_lap = new Float64Array(this.width * this.height);

        // Some precomputations
        this.init();
    }

    init() {
        // seed the V state with perlin noise and the U state with 1s.
        noise.seed(Math.random());
        for (let i = 0; i < this.width; i++) {
            for (let j = 0; j < this.height; j++) {
                this.U[this.width * j + i] = 1.;
                let n = 0, freq = 0, max = 0;
                // fractal brownian motion
                for (let o = 0; o < 21; o++) {
                    freq = Math.pow(2, o);
                    max += (1 / freq);
                    n += noise.simplex2(
                        i * freq * this.height / 8,
                        j * freq * this.width / 8) / freq;
                }
                n /= max;
                // these parameters are very sensitive to f and k
                if (n > 0.525) {
                    this.V[this.width * j + i] = (1.5 + n) / 4.;
                } else {
                    this.V[this.width * j + i] = 0;
                }
            }
        }
        // init drawing buffer to black
        for (let idx = 0, i = 0; idx < this.buffer.length; idx += 4, i++) {
            this.buffer[idx    ] = 0;
            this.buffer[idx + 1] = 0;
            this.buffer[idx + 2] = 0;
            this.buffer[idx + 3] = 255;
        }
    }

    /* Return the f and k parameters for the current phase.
     *
     * @param {Number} t
     */
    fk(t) {
        t = (t % this.PERIOD) / this.PERIOD;
        let t_ = this.phase_freq;
        for (let i = 0; i < this.num_phases; i++) {
            if (t_ > t)  {
                return [
                    this.fs[i], 
                    this.ks[i]
                ];
            }
            t_ += this.phase_freq;
        }
    }

    /* This is for interaction. Adding this to a click handler allows clicks
     * to drop in some V chemical.
     *
     * @param {Integer} i
     * @param {Integer} j
     */
    drop(i, j) {
        this.V[this.width * j + i] = 0.75;
    }

    /* Run the simulation for one timestep with parameters f, k. We ignore the
     * length of the timestep.
     *
     * @param {Number} f
     * @param {Number} k
     */
    step(f, k) {
        this.toroidalLaplacian2D(this.U, this.U_lap);
        this.toroidalLaplacian2D(this.V, this.V_lap);
        let reaction_rate = 0;
        for (let idx = 0; idx < this.width * this.height; idx++) {
            reaction_rate = this.U[idx] * this.V[idx] * this.V[idx];
            this.U[idx] = this.U[idx] + this.D_u * this.U_lap[idx] - reaction_rate + f * (1.0 - this.U[idx]);
            this.V[idx] = this.V[idx] + this.D_v * this.V_lap[idx] + reaction_rate - (k + f) * this.V[idx];
        }
    }


    /* Map U and V onto colors and place them in a buffer. Then blit that buffer
     * onto the canvas.
     */
    draw() {
        let eu = 0, ev = 0, evu = 0;
        for (let idx = 0, i = 0; idx < this.buffer.length; idx += 4, i++) {
            eu = Math.exp(1.5 * (0.5 + this.U[i]));
            ev = Math.exp(5. * this.V[i]);
            this.buffer[idx + 0] = Math.floor(255 * ev / (ev + 1));
            this.buffer[idx + 1] = Math.floor(255 * (0.5 * ev * eu / (0.5 * ev * eu + 1)));
            this.buffer[idx + 2] = Math.floor(255 * eu / (eu + 1));
        }
        this.context.putImageData(this.id, 0, 0);
    }

    /**
     * Get the 2D Laplacian of an Array-like using a convolution, and save the
     * result in another Array-like buffer. Conv padding style is `wrap`, kind
     * of unrolled instead of actually padding.
     *
     * @param {Array-like} input
     * @param {Array-like} kernel
     * @param {Array-like} result
     */
    toroidalLaplacian2D(input, result) {
        const w = this.width;
        const h = this.height;

        /* Laplacian can be approximated by convolution against the kernel
         *   0.05  0.2  0.05
         *   0.2  -1.0  0.2
         *   0.05  0.2  0.05
         */

        // idx will always equal `w * j + i`
        let idx = 0;

        // these hold the values of input at all eight neighbor positions
        // of idx.
        let u = 0, r = 0, d = 0, l = 0,
            ul = 0, ur = 0, bl = 0, br = 0;

        // loop through by columns and then rows for caching reasons (idx ordering)
        // cases are handled inside this giant loop rather than outside, since
        // this seems fastest after some profiling.
        for (let j = 0; j < h; j++) {
            for (let i = 0; i < w; i++, idx++) {

                // upper neighbor
                if (j == 0) {
                    u = input[w * (h - 1) + i];
                } else {
                    u = input[idx - w];
                }

                // upper right
                if (j == 0) {
                    if (i == w - 1) {
                        ur = input[w * (h - 1)];
                    } else {
                        ur = input[w * (h - 1) + i + 1];
                    }
                } else {
                    if (i == w - 1) {
                        ur = input[w * (j - 1)];
                    } else {
                        ur = input[idx - w + 1];
                    }
                }

                // right
                if (i == w - 1) {
                    r = input[w * j];
                } else {
                    r = input[idx + 1];
                }

                // bottom right
                if (j == h - 1) {
                    if (i == w - 1) {
                        br = input[0];
                    } else {
                        br = input[i + 1];
                    }
                } else {
                    if (i == w - 1) {
                        br = input[w * (j + 1)];
                    } else {
                        br = input[idx + w + 1];
                    }
                }
                
                // down
                if (j == h - 1) {
                    d = input[i];
                } else {
                    d = input[idx + w];
                }

                // bottom left
                if (j == h - 1) {
                    if (i == 0) {
                        bl = input[w - 1];
                    } else {
                        bl = input[i - 1];
                    }
                } else {
                    if (i == 0) {
                        bl = input[w * (j + 2) - 1];
                    } else {
                        bl = input[idx + w - 1];
                    }
                }

                // left
                if (i == 0) {
                    l = input[idx + w - 1];
                } else {
                    l = input[idx - 1];
                }

                // upper left
                if (j == 0) {
                    if (i == 0) {
                        ul = input[w * h - 1];
                    } else {
                        ul = input[w * (h - 1) + i - 1];
                    }
                } else {
                    if (i == 0) {
                        ul = input[w * j - 1];
                    } else {
                        ul = input[idx - w - 1];
                    }
                }
                
                // finally, compute this cell's portion of the convolution
                // described above.
                result[idx] = 0.2 * (u + r + d + l) 
                            + 0.05 * (ul + ur + bl + br) 
                            - input[idx];
            }
        }
    }
}

/* main ******************************************************************** */
window.onload = function() {
    let canvas = document.getElementById('cvs');
    let w = canvas.offsetWidth;
    let h = canvas.offsetHeight;
    // Size of the reaction. 100 seems not too crazy for my phone so might
    // be ok for everyone.
    let S = 100;
    if (w > h) {
        canvas.setAttribute('width', S);
        canvas.setAttribute('height', Math.floor(S * h / w));
    } else {
        canvas.setAttribute('height', S);
        canvas.setAttribute('width', Math.floor(S * w / h));
    }
    let reaction = new GrayScottRxnDfn(canvas);
    let fk = reaction.fk(0);
    let mousedown = false;
    // step a few times to clear out the noisy IC.
    reaction.step(fk[0], fk[1]);
    reaction.step(fk[0], fk[1]);
    reaction.step(fk[0], fk[1]);
    reaction.step(fk[0], fk[1]);
    reaction.step(fk[0], fk[1]);

    // This is the main loop.
    function loop(t) {
        fk = reaction.fk(t);
        reaction.step(fk[0], fk[1]);
        reaction.draw();
        window.requestAnimationFrame(loop);
    }

    // some event handlers for interaction with GrayScottRxnDfn.drop(...).
    function start(e) {
        mousedown = true;
        clicker(e);
    }
    function clicker(e) {
        if (mousedown) {
            const i = Math.ceil(S * e.clientX / canvas.offsetWidth);
            const j = Math.ceil(S * e.clientY / canvas.offsetWidth);
            reaction.drop(i, j);
        }
    }
    function end() {
        mousedown = false;
    }
    document.addEventListener('mousedown', start);
    document.addEventListener('touchstart', start);
    document.addEventListener('mouseup', end);
    document.addEventListener('touchcancel', end);
    document.addEventListener('touchend', end);
    document.addEventListener('mousemove', clicker);
    document.addEventListener('touchmove', clicker);

    // run it.
    window.requestAnimationFrame(loop);
}

