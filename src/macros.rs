/// Retire a variable by shadowing it out of scope so that it can't be reused.
#[allow(unused)]
macro_rules! retire {
    ($name:ident) => {
        #[allow(unused_variables)]
        let $name = ();
    };
}
