import _ from 'lodash'

//  Test functions
// https://en.wikipedia.org/wiki/Test_functions_for_optimization

// sphere func
function func1(x) {
    return x.reduce((prev, current) => {
        return prev + Math.pow(current, 2);
    }, 0)
}

// Himmelblau's function
function func2(x) {
    const e = Math.pow(Math.pow(x[0], 2) + x[1]-11, 2) + Math.pow(x[0] + Math.pow(x[1], 2) -7, 2)
    return e
}

// func1(_.range(12)).then(a => console.log(a)).catch();


function ensure_bounds(vec, bounds) {
    const vec_new = []
    for (let i of _.range(vec.length)) {
        // console.log(vec)
        // console.log(bounds)

        if (vec[i] < bounds[i][0]) {
            vec_new.push(bounds[i][0])
        }
    
        if (vec[i] > bounds[i][1]) {
            vec_new.push(bounds[i][1])
        }
    
        if (bounds[i][0] <= vec[i] <= bounds[i][1]) {
            vec_new.push(vec[i])
        }
    }
 
    return vec_new
}

function main(cost_func, bounds, popsize, mutate, recombination, maxiter) {

    //--- INITIALIZE A POPULATION (step #1) ----------------+
    
    const population = []
    for (let i in _.range(popsize)) {
        let indv = []
        for (let j in _.range(bounds.length)) {
            // indv.push(_.random(bounds[j][0],bounds[j][1]))
            // indv.push(_.random(bounds[j][0],bounds[j][1], true))
            indv.push(bounds[j][0] + Math.random()*(bounds[j][1] - bounds[j][0]))
        }
        population.push(indv)
    }

    console.log('first pop', population)
            
    //--- SOLVE --------------------------------------------+
    
    // cycle through each generation (step #2)
    for (let i in _.range(maxiter+1)) {
        console.log('GENERATION:', i)
    
        var gen_scores = [] // score keeping
    
        // cycle through each individual in the population
        for (let j in _.range(popsize)) {
            //--- MUTATION (step #3.A) ---------------------+
            
            // select three random vector index positions [0, popsize), not including current vector (j)
            let canidates = _.range(popsize)
            canidates.splice(j, 1)
            let random_index = _.sampleSize(canidates, 3)
    
            let x_1 = population[random_index[0]]
            let x_2 = population[random_index[1]]
            let x_3 = population[random_index[2]]
            let x_t = population[j]     // target individual
    
            // subtract x3 from x2, and create a new vector (x_diff)
            let x_diff = _.zip(x_3, x_2).map(e => e[0] - e[1])
    
            // multiply x_diff by the mutation factor (F) and add to x_1
            let v_donor = _.zip(x_1, x_diff).map(e => e[0] + mutate * e[1])
            v_donor = ensure_bounds(v_donor, bounds)
    
            //--- RECOMBINATION (step #3.B) ----------------+
    
            let v_trial = []
            for (let k in _.range(x_t.length)) {
                let crossover = Math.random()
                if (crossover <= recombination) {
                    v_trial.push(v_donor[k])
                    
                } else {
                    v_trial.push(x_t[k])
                }
            }
                    
            //--- GREEDY SELECTION (step #3.C) -------------+
    
            let score_trial  = cost_func(v_trial)
            let score_target = cost_func(x_t)
    
            if (score_trial < score_target) {
                population[j] = v_trial
                gen_scores.push(score_trial)
                console.log('   >',score_trial, v_trial)
            } else {
                console.log( '   >',score_target, x_t)
                gen_scores.push(score_target)
            }
        }
    
    
    }
    //--- SCORE KEEPING --------------------------------+

    let gen_avg = _.sum(gen_scores) / popsize                         // current generation avg. fitness
    let gen_best = _.min(gen_scores)                                  // fitness of best individual
    let gen_sol = population[gen_scores.indexOf(gen_best)]     // solution of best individual

    console.log( ' > GENERATION AVERAGE:',gen_avg)
    console.log( ' > GENERATION BEST:',gen_best)
    console.log( ' > BEST SOLUTION:',gen_sol)
    
    return gen_sol
}


//--- CONSTANTS ----------------------------------------------------------------+

const cost_func = func2                   // Cost function
const bounds = [[-5,5],[-5,5]]            // Bounds [(x1_min, x1_max), (x2_min, x2_max),...]
const popsize = 20                        // Population size, must be >= 4
const mutate = 0.5                        // Mutation factor [0,2]
const recombination = 0.7                 // Recombination rate [0,1]
const maxiter = 40                        // Max number of generations (maxiter)

//--- RUN ----------------------------------------------------------------------+

main(cost_func, bounds, popsize, mutate, recombination, maxiter)

//--- END ----------------------------------------------------------------------+

