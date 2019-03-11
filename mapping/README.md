## Mappings

### What are these

Here we provide three mappings that maps FreeBase types to different taxonomies.

### How to modify or create your own

Each mapping is composed with two files: *.mapping and *.logic.mapping

Each line in a *.mapping file contains a tab-separated pair, where the left is a FreeBase type, and the right is a target type.
Note that each target type is automatically dissected into hierarchies by splitting "/". This file works in
a "OR" logic, that is, a target type is assigned if *any* of the FreeBase mapping sources is found in an entity's FreeBase types.

*.logic.mapping serves as a supplementary mapping for more logics beyond "OR". 
Each lineThere are three patterns of each line:
- "+\tA\tB" means if type A appears in the mapped types, type B will be added
- "-\tA\tB" means if type A appears in the mapped types, type B (if present) will be removed
- "=\tA\tB" means if type A and type B are equivalent for evaluation purposes.

