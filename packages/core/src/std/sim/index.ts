/**
 * std/sim — the simulation vocabulary (compute-backed state), grouped by family. The grid
 * simulation nouns (`simulate`, `op`, `GridSim`) stay top-level on `shaders/std`; these are the
 * L1 words the agent, fluid, feedback and grid simulations are built from.
 */
export * as agents from './agents'
export * as agentForces from './agentForces'
export * as agentFrame from './agentFrame'
export * as agentRender from './agentRender'
export * as feedback from './feedback'
export * as fluids from './fluids'
export * as grids from './grids'
export * as shapeFields from './shapeFields'
