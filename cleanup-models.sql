-- delete all data for model, transect pairs for which all values are null

with
nonnull_counts as (
    select model_id, transect_id, sum(iif(x is null, 0, 1)) as n_nonnull
    from data
    group by model_id, transect_id
),
all_null as (
    select model_id, transect_id
    from nonnull_counts
    where n_nonnull = 0
)

delete from data as d
where exists(
    select 1
    from all_null a
    where d.model_id = a.model_id and d.transect_id = a.transect_id
);

--select model.name, count(*) as n
--from all_null
--    join model on model_id = model.id
--group by model.name;